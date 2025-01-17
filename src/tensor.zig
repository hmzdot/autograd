const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Tensor over generic T
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        data: []T,
        size: []usize,
        stride: []usize,

        /// Builds stride from size
        inline fn buildStride(size: []const usize, stride: []usize) void {
            // `data` is organized so that values in the innermost layer is
            // adjacent. With every layer on top, we have to multiply by the
            // size of the layer.
            // For example if the shape of the tensor is (3,4,5), this loop
            // will set a stride (1,3,12).
            var running_stride: usize = 1;
            var i: usize = size.len;
            while (i > 0) {
                i -= 1;
                stride[i] = running_stride;
                running_stride *= size[i];
            }
        }

        /// Initialize with owned data
        pub fn initFromOwned(size: []usize, data: []T, allocator: Allocator) !Tensor(T) {
            var size_total: usize = 1;
            for (size) |s| {
                size_total *= s;
            }
            std.debug.assert(size_total == data.len);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size, stride_heap);

            return Self{
                .allocator = allocator,
                .data = data,
                .size = size,
                .stride = stride_heap,
            };
        }

        /// Initialize with a slice
        pub fn initFromSlice(size: []const usize, data: []const T, allocator: Allocator) !Tensor(T) {
            var size_total: usize = 1;
            for (size) |s| {
                size_total *= s;
            }
            std.debug.assert(size_total == data.len);

            const data_heap = try allocator.alloc(T, data.len);
            @memcpy(data_heap, data);

            const size_heap = try allocator.alloc(usize, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size_heap, stride_heap);

            return Self{
                .allocator = allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
            };
        }

        pub fn deinit(self: *Tensor(T)) void {
            self.allocator.free(self.data);
            self.allocator.free(self.size);
            self.allocator.free(self.stride);
        }

        /// Print data
        pub fn print(self: *Tensor(T)) void {
            for (self.data) |d| {
                std.debug.print("{} ", .{d});
            }
            std.debug.print("\n", .{});
        }
    };
}

pub fn add(comptime T: type, a: *const Tensor(T), b: *const Tensor(T)) !Tensor(T) {
    const same_size = for (a.size, b.size) |ad, bd| {
        if (ad != bd) break false;
    } else true;
    std.debug.assert(same_size);

    const size: []usize = try a.allocator.alloc(usize, a.size.len);
    @memcpy(size, a.size);

    var c_data: []T = try a.allocator.alloc(T, a.data.len);
    for (a.data, b.data, 0..) |ai, bi, i| {
        c_data[i] = ai + bi;
    }
    return Tensor(T).initFromOwned(size, c_data, a.allocator);
}

pub fn mul(comptime T: type, a: *const Tensor(T), b: *const Tensor(T)) !Tensor(T) {
    const a_end = a.size.len - 1;
    std.debug.assert(a.size[a_end] == b.size[0]);

    const shared_dim = a.size[a_end];

    // Reshape A and B to (-1, v_size)
    // NOTE: This will change when a.data.len != total size of a
    const a_dim = a.data.len / shared_dim;
    const b_dim = b.data.len / shared_dim;
    const a_stride: [2]usize = .{ shared_dim, 1 };
    const b_stride: [2]usize = .{ b_dim, 1 };

    // Perform 2x2 matrix multiplication
    var c_data = try a.allocator.alloc(T, a_dim * b_dim);
    for (0..a_dim) |ad| {
        for (0..b_dim) |bd| {
            const a_offset = a_stride[0] * ad;
            const b_offset = b_stride[1] * bd;

            var sum: T = 0;
            for (0..shared_dim) |vi| {
                sum += a.data[a_offset + vi * a_stride[1]] * b.data[b_offset + vi * b_stride[0]];
            }
            c_data[ad * b_dim + bd] = sum;
        }
    }

    var size = try a.allocator.alloc(usize, a.size.len + b.size.len - 2);
    @memcpy(size[0..a_end], a.size[0..a_end]);
    @memcpy(size[a_end..], b.size[1..]);

    return Tensor(T).initFromOwned(size, c_data, a.allocator);
}

test "Tensor::initFromOwned" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f32, 6);
    for (data, 0..) |*d, i| d.* = @floatFromInt(i);

    const size = try allocator.alloc(usize, 2);
    @memcpy(size, &[_]usize{ 2, 3 });

    var t = try Tensor(f32).initFromOwned(size, data, allocator);
    defer t.deinit();

    try testing.expectEqual(@as(usize, 2), t.size[0]);
    try testing.expectEqual(@as(usize, 3), t.size[1]);
    try testing.expectEqualSlices(f32, data, t.data);
}

test "Tensor::initFromSlice" {
    const allocator = testing.allocator;
    const data = [_]f32{ 0, 1, 2, 3, 4, 5 };
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer t.deinit();

    try testing.expectEqual(@as(usize, 2), t.size[0]);
    try testing.expectEqual(@as(usize, 3), t.size[1]);
    try testing.expectEqualSlices(f32, &data, t.data);
}

test "add" {
    const allocator = testing.allocator;
    const data_a = [_]f32{ 1, 2, 3, 4 };
    const data_b = [_]f32{ 5, 6, 7, 8 };
    const size = [_]usize{ 2, 2 };

    var a = try Tensor(f32).initFromSlice(&size, &data_a, allocator);
    defer a.deinit();
    var b = try Tensor(f32).initFromSlice(&size, &data_b, allocator);
    defer b.deinit();

    var c = try add(f32, &a, &b);
    defer c.deinit();

    const expected = [_]f32{ 6, 8, 10, 12 };
    try testing.expectEqualSlices(f32, &expected, c.data);
}

test "mul" {
    const allocator = testing.allocator;

    // Test case for 2x3 * 3x2 matrix multiplication
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 2x3 matrix
    const size_a = [_]usize{ 2, 3 };
    const data_b = [_]f32{ 7, 8, 9, 10, 11, 12 }; // 3x2 matrix
    const size_b = [_]usize{ 3, 2 };

    var a = try Tensor(f32).initFromSlice(&size_a, &data_a, allocator);
    defer a.deinit();
    var b = try Tensor(f32).initFromSlice(&size_b, &data_b, allocator);
    defer b.deinit();

    var c = try mul(f32, &a, &b);
    defer c.deinit();

    // Expected result should be a 2x2 matrix:
    // [1 2 3]   [7  8 ]   [58  64]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    const expected = [_]f32{ 58, 64, 139, 154 };

    // Check dimensions
    try testing.expectEqual(@as(usize, 2), c.size[0]);
    try testing.expectEqual(@as(usize, 2), c.size[1]);

    // Check values
    try testing.expectEqualSlices(f32, &expected, c.data);
}
