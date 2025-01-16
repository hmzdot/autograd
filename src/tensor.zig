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
        pub fn initFromOwned(size: []const usize, data: []T, allocator: Allocator) !Tensor(T) {
            var size_total: usize = 1;
            for (size) |s| {
                size_total *= s;
            }
            std.debug.assert(size_total == data.len);

            const size_heap = try allocator.alloc(usize, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size_heap, stride_heap);

            return Self{
                .allocator = allocator,
                .data = data,
                .size = size_heap,
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

fn add(comptime T: type, a: *const Tensor(T), b: *const Tensor(T)) !Tensor(T) {
    const same_size = for (a.size, b.size) |ad, bd| {
        if (ad != bd) break false;
    } else true;
    std.debug.assert(same_size);

    var c_data: []T = try a.allocator.alloc(T, a.data.len);
    for (a.data, b.data, 0..) |ai, bi, i| {
        c_data[i] = ai + bi;
    }
    return Tensor(T).initFromOwned(a.size, c_data, a.allocator);
}

test "Tensor::initFromOwned" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f32, 6);
    for (data, 0..) |*d, i| d.* = @floatFromInt(i);

    const size = [_]usize{ 2, 3 };
    var t = try Tensor(f32).initFromOwned(&size, data, allocator);
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
