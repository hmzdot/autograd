const std = @import("std");
const arc = @import("./arc.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;
const Arc = arc.Arc;

// TODO: Consider the empty tensor
// TODO: Add refcount

const Op = enum { add, mul, none };

fn printSlice(comptime T: type, slice: []const T) void {
    std.debug.print("[", .{});
    for (slice[0..(slice.len - 1)]) |s| {
        std.debug.print("{},", .{s});
    }
    if (slice.len > 0) {
        std.debug.print("{}", .{slice[slice.len - 1]});
    }
    std.debug.print("]\n", .{});
}

// Tensor iterator that respects stride
fn Iterator(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        stride: []const usize,
        size: []const usize,
        index: usize,
        total: usize,

        pub fn fromTensor(tensor: *const Tensor(T)) Self {
            var total: usize = 1;
            for (tensor.size) |s| total *= s;

            return Iterator(T){
                .data = tensor.data,
                .stride = tensor.stride,
                .size = tensor.size,
                .index = 0,
                .total = total,
            };
        }

        pub fn next(self: *Self) ?T {
            if (self.index >= self.total) {
                return null;
            } else {
                var offset: usize = 0;
                var rem_index = self.index;
                var dim_index = self.size.len;
                while (dim_index > 0) : (dim_index -= 1) {
                    const st = self.stride[dim_index - 1];
                    const sz = self.size[dim_index - 1];
                    offset += (rem_index % sz) * st;
                    rem_index /= sz;
                }
                self.index += 1;
                return self.data[offset];
            }
        }
    };
}

fn Graph(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        operation: Op,
        operands: ?[]*Tensor(T),

        pub fn init(op: Op, operands: []const *Tensor(T), allocator: Allocator) !Graph(T) {
            const operands_heap = try allocator.alloc(*Tensor(T), operands.len);
            @memcpy(operands_heap, operands[0..]);

            return Self{
                .allocator = allocator,
                .operation = op,
                .operands = operands_heap,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.operands) |ops| self.allocator.free(ops);
            self.allocator.destroy(self);
        }
    };
}

/// Tensor over generic T
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        gradient: ?[]T,
        comp_graph: ?*Graph(T),

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
                .gradient = null,
                .comp_graph = null,
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
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn clone(self: *const Self) !Tensor(T) {
            const data_heap = try self.allocator.alloc(T, self.data.len);
            @memcpy(data_heap, self.data);

            const size_heap = try self.allocator.alloc(usize, self.size.len);
            @memcpy(size_heap, self.size);

            const stride_heap = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_heap, self.stride);

            return Self{
                .allocator = self.allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn ones(size: []const usize, allocator: Allocator) !Tensor(T) {
            var num_elements: usize = 1;
            for (size) |s| num_elements *= s;

            const data_heap = try allocator.alloc(T, num_elements);
            for (0..num_elements) |i| {
                data_heap[i] = 1;
            }

            const size_heap = try allocator.alloc(usize, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size_heap, stride_heap);

            return Self{
                .allocator = allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.size);
            self.allocator.free(self.stride);
            if (self.comp_graph) |cg| cg.deinit();
            if (self.gradient) |grad| self.allocator.free(grad);
        }

        pub fn transpose(
            self: *const Self,
            dim0: usize,
            dim1: usize,
        ) !Tensor(T) {
            if (dim0 >= self.size.len or dim1 >= self.size.len) {
                return error.InvalidDimensions;
            }

            // TODO: Copying data here makes everything pointless, but currently
            // there's no mechanism to refcount.
            const new_data = try self.allocator.alloc(T, self.data.len);
            @memcpy(new_data, self.data);

            var new_size = try self.allocator.alloc(usize, self.size.len);
            var new_stride = try self.allocator.alloc(usize, self.stride.len);

            // Copy the original sizes and strides into the new ones
            for (self.size, 0..) |size, i| {
                new_size[i] = size;
            }
            for (self.stride, 0..) |stride, i| {
                new_stride[i] = stride;
            }

            // Swap the dimensions
            std.mem.swap(usize, &new_size[dim0], &new_size[dim1]);
            std.mem.swap(usize, &new_stride[dim0], &new_stride[dim1]);

            return Tensor(T){
                .allocator = self.allocator,
                .gradient = null,
                .comp_graph = null,
                .data = new_data,
                .size = new_size,
                .stride = new_stride,
            };
        }

        pub fn is_contiguous(self: *const Self) bool {
            const stride_len = self.stride.len;
            if (stride_len == 1) return true;

            for (0..(self.stride.len - 1)) |i| {
                if (self.stride[i] < self.stride[i + 1]) {
                    return false;
                }
            }
            return true;
        }

        pub fn contiguous(self: *Self) !void {
            var data = try self.allocator.alloc(T, self.data.len);
            var it = Iterator(T).fromTensor(self);

            var next = it.next();
            while (next != null) : (next = it.next()) {
                data[it.index - 1] = next.?;
            }

            // Replace data with contiguous one
            self.allocator.free(self.data);
            self.data = data;

            // Build stride from scratch, as it might be invalidated
            buildStride(self.size, self.stride);
        }

        // Returns total number of items
        pub fn numel(self: *Self) usize {
            var n: usize = 1;
            for (self.size) |s| n *= s;
            return n;
        }

        fn is_reshape_valid(old: []const usize, new: []const usize) bool {
            var old_size: usize = 1;
            for (old) |o| old_size *= o;

            var new_size: usize = 1;
            for (new) |n| new_size *= n;

            return old_size == new_size;
        }

        pub fn resize(self: *Self, new_size: []const usize) !Self {
            if (!is_reshape_valid(self.size, new_size)) {
                return error.IncompatibleSize;
            }
            var tensor = try self.clone();
            if (!tensor.is_contiguous()) try tensor.contiguous();

            @memcpy(tensor.size, new_size);
            buildStride(tensor.size, tensor.stride);
            return tensor;
        }

        /// Print data
        pub fn print(self: *Self) void {
            var it = Iterator(T).fromTensor(self);
            var next = it.next();
            while (next != null) : (next = it.next()) {
                std.debug.print("{} ", .{next.?});
            }
            std.debug.print("\n", .{});
        }

        // Convenience functions

        pub fn add(self: *Self, other: *Self) !Self {
            return _add(T, self, other);
        }

        pub fn matmul(self: *Self, other: *Self) !Self {
            return _matmul(T, self, other);
        }

        pub fn backward(self: *Self) !void {
            try _backward(T, self);
        }
    };
}

/// Tensor over generic T
pub fn TensorArc(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        gradient: ?[]T,
        comp_graph: ?*Graph(T),

        data: Arc([]T),
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
        pub fn initFromOwned(size: []usize, data: []T, allocator: Allocator) !Self {
            var size_total: usize = 1;
            for (size) |s| {
                size_total *= s;
            }
            std.debug.assert(size_total == data.len);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size, stride_heap);

            const arc_data = Arc([]T).init(data, Self.deinit_data, allocator);

            return Self{
                .allocator = allocator,
                .data = arc_data,
                .size = size,
                .stride = stride_heap,
                .gradient = null,
                .comp_graph = null,
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
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn clone(self: *const Self) !Tensor(T) {
            const data_heap = try self.allocator.alloc(T, self.data.len);
            @memcpy(data_heap, self.data);

            const size_heap = try self.allocator.alloc(usize, self.size.len);
            @memcpy(size_heap, self.size);

            const stride_heap = try self.allocator.alloc(usize, self.stride.len);
            @memcpy(stride_heap, self.stride);

            return Self{
                .allocator = self.allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn ones(size: []const usize, allocator: Allocator) !Tensor(T) {
            var num_elements: usize = 1;
            for (size) |s| num_elements *= s;

            const data_heap = try allocator.alloc(T, num_elements);
            for (0..num_elements) |i| {
                data_heap[i] = 1;
            }

            const size_heap = try allocator.alloc(usize, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size_heap, stride_heap);

            return Self{
                .allocator = allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
                .gradient = null,
                .comp_graph = null,
            };
        }

        pub fn deinit_data(data: []T, allocator: Allocator) void {
            allocator.free(data);
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.size);
            self.allocator.free(self.stride);
            if (self.comp_graph) |cg| cg.deinit();
            if (self.gradient) |grad| self.allocator.free(grad);
            self.data.unref();
        }

        pub fn transpose(
            self: *const Self,
            dim0: usize,
            dim1: usize,
        ) !Tensor(T) {
            if (dim0 >= self.size.len or dim1 >= self.size.len) {
                return error.InvalidDimensions;
            }

            // TODO: Copying data here makes everything pointless, but currently
            // there's no mechanism to refcount.
            const new_data = try self.allocator.alloc(T, self.data.len);
            @memcpy(new_data, self.data);

            var new_size = try self.allocator.alloc(usize, self.size.len);
            var new_stride = try self.allocator.alloc(usize, self.stride.len);

            // Copy the original sizes and strides into the new ones
            for (self.size, 0..) |size, i| {
                new_size[i] = size;
            }
            for (self.stride, 0..) |stride, i| {
                new_stride[i] = stride;
            }

            // Swap the dimensions
            std.mem.swap(usize, &new_size[dim0], &new_size[dim1]);
            std.mem.swap(usize, &new_stride[dim0], &new_stride[dim1]);

            return Tensor(T){
                .allocator = self.allocator,
                .gradient = null,
                .comp_graph = null,
                .data = new_data,
                .size = new_size,
                .stride = new_stride,
            };
        }

        pub fn is_contiguous(self: *const Self) bool {
            const stride_len = self.stride.len;
            if (stride_len == 1) return true;

            for (0..(self.stride.len - 1)) |i| {
                if (self.stride[i] < self.stride[i + 1]) {
                    return false;
                }
            }
            return true;
        }

        pub fn contiguous(self: *Self) !void {
            var data = try self.allocator.alloc(T, self.data.len);
            var it = Iterator(T).fromTensor(self);

            var next = it.next();
            while (next != null) : (next = it.next()) {
                data[it.index - 1] = next.?;
            }

            // Replace data with contiguous one
            self.allocator.free(self.data);
            self.data = data;

            // Build stride from scratch, as it might be invalidated
            buildStride(self.size, self.stride);
        }

        // Returns total number of items
        pub fn numel(self: *Self) usize {
            var n: usize = 1;
            for (self.size) |s| n *= s;
            return n;
        }

        fn is_reshape_valid(old: []const usize, new: []const usize) bool {
            var old_size: usize = 1;
            for (old) |o| old_size *= o;

            var new_size: usize = 1;
            for (new) |n| new_size *= n;

            return old_size == new_size;
        }

        pub fn resize(self: *Self, new_size: []const usize) !Self {
            if (!is_reshape_valid(self.size, new_size)) {
                return error.IncompatibleSize;
            }
            var tensor = try self.clone();
            if (!tensor.is_contiguous()) try tensor.contiguous();

            @memcpy(tensor.size, new_size);
            buildStride(tensor.size, tensor.stride);
            return tensor;
        }

        /// Print data
        pub fn print(self: *Self) void {
            var it = Iterator(T).fromTensor(self);
            var next = it.next();
            while (next != null) : (next = it.next()) {
                std.debug.print("{} ", .{next.?});
            }
            std.debug.print("\n", .{});
        }

        // Convenience functions

        pub fn add(self: *Self, other: *Self) !Self {
            return _add(T, self, other);
        }

        pub fn matmul(self: *Self, other: *Self) !Self {
            return _matmul(T, self, other);
        }

        pub fn backward(self: *Self) !void {
            try _backward(T, self);
        }
    };
}

pub fn _add(comptime T: type, a: *Tensor(T), b: *Tensor(T)) !Tensor(T) {
    const same_size = for (a.size, b.size) |ad, bd| {
        if (ad != bd) break false;
    } else true;
    std.debug.assert(same_size);

    const size: []usize = try a.allocator.alloc(usize, a.size.len);
    @memcpy(size, a.size);

    var data: []T = try a.allocator.alloc(T, a.data.len);
    var a_it = Iterator(T).fromTensor(a);
    var b_it = Iterator(T).fromTensor(b);
    while (true) {
        const a_next = a_it.next() orelse break;
        const b_next = b_it.next() orelse break;
        data[a_it.index - 1] = a_next + b_next;
    }

    var t = try Tensor(T).initFromOwned(size, data, a.allocator);
    t.comp_graph = try a.allocator.create(Graph(T));
    t.comp_graph.?.* = try Graph(T).init(Op.add, &.{ a, b }, a.allocator);
    return t;
}

pub fn _matmul(comptime T: type, a: *Tensor(T), b: *Tensor(T)) !Tensor(T) {
    const a_end = a.size.len - 1;
    std.debug.assert(a.size[a_end] == b.size[0]);

    const shared_dim = a.size[a_end];

    // Reshape A and B to (-1, v_size)
    const a_dim: usize = a.numel() / shared_dim;
    var a_2d = try a.resize(&.{ a_dim, shared_dim });
    defer a_2d.deinit();

    const b_dim: usize = b.numel() / shared_dim;
    var b_2d = try b.resize(&.{ shared_dim, b_dim });
    defer b_2d.deinit();

    // Perform 2x2 matrix multiplication
    var c_data = try a.allocator.alloc(T, a_dim * b_dim);
    for (0..a_dim) |ad| {
        for (0..b_dim) |bd| {
            const a_offset = a_2d.stride[0] * ad;
            const b_offset = b_2d.stride[1] * bd;

            var sum: T = 0;
            for (0..shared_dim) |vi| {
                sum += a_2d.data[a_offset + vi * a_2d.stride[1]] * b_2d.data[b_offset + vi * b_2d.stride[0]];
            }
            c_data[ad * b_dim + bd] = sum;
        }
    }

    var size = try a.allocator.alloc(usize, a.size.len + b.size.len - 2);
    @memcpy(size[0..a_end], a.size[0..a_end]);
    @memcpy(size[a_end..], b.size[1..]);

    var t = try Tensor(T).initFromOwned(size, c_data, a.allocator);
    t.comp_graph = try a.allocator.create(Graph(T));
    t.comp_graph.?.* = try Graph(T).init(Op.mul, &.{ a, b }, a.allocator);
    return t;
}

pub fn _backward(comptime T: type, t: *Tensor(T)) !void {
    if (t.gradient == null) {
        t.gradient = try t.allocator.alloc(T, t.data.len);
        var i: usize = 0;
        while (i < t.data.len) : (i += 1) {
            t.gradient.?[i] = 1;
        }
    }

    if (t.comp_graph) |cg| {
        switch (cg.operation) {
            .add => {
                const op0 = cg.operands.?[0]; // left operand
                if (op0.gradient == null) {
                    op0.gradient = try op0.allocator.alloc(T, op0.data.len);
                    @memset(op0.gradient.?, 0);
                }

                const op1 = cg.operands.?[1]; // right operand
                if (op1.gradient == null) {
                    op1.gradient = try op1.allocator.alloc(T, op1.data.len);
                    @memset(op1.gradient.?, 0);
                }

                // propagate gradients to parents
                const grad0 = op0.gradient.?;
                const grad1 = op1.gradient.?;
                var i: usize = 0;
                while (i < t.data.len) : (i += 1) {
                    grad0[i] += t.gradient.?[i];
                    grad1[i] += t.gradient.?[i];
                }

                try _backward(T, op0);
                try _backward(T, op1);
            },
            .mul => {
                const op0 = cg.operands.?[0]; // left operand
                if (op0.gradient == null) {
                    op0.gradient = try op0.allocator.alloc(T, op0.data.len);
                    @memset(op0.gradient.?, 0);
                }

                const op1 = cg.operands.?[1]; // right operand
                if (op1.gradient == null) {
                    op1.gradient = try op1.allocator.alloc(T, op1.data.len);
                    @memset(op1.gradient.?, 0);
                }

                var grad_in = try Tensor(T).initFromSlice(
                    op0.size,
                    t.gradient.?,
                    op0.allocator,
                );

                // ∂C/∂A = ∂C · Bᵀ
                var transposed_op1 = try op1.transpose(0, 1);
                var grad0_result = try grad_in.matmul(&transposed_op1);

                // ∂C/∂B = Aᵀ · ∂C
                var transposed_op0 = try op0.transpose(0, 1);
                var grad1_result = try transposed_op0.matmul(&grad_in);

                // accumulate gradients
                for (op0.gradient.?, grad0_result.data) |*grad, data| {
                    grad.* += data;
                }
                for (op1.gradient.?, grad1_result.data) |*grad, data| {
                    grad.* += data;
                }

                // clean up temporary tensors
                grad0_result.deinit();
                grad1_result.deinit();
                transposed_op1.deinit();
                transposed_op0.deinit();
                grad_in.deinit();

                try _backward(T, op0);
                try _backward(T, op1);
            },
            .none => return,
        }
    }
}

test "Tensor::initFromOwned" {
    const allocator = testing.allocator;
    const data = try allocator.alloc(f32, 6);
    for (data, 0..) |*d, i| d.* = @floatFromInt(i);

    const size = try allocator.alloc(usize, 2);
    @memcpy(size, &[_]usize{ 2, 3 });

    var t = try TensorArc(f32).initFromOwned(size, data, allocator);
    defer t.deinit();

    try testing.expectEqual(@as(usize, 2), t.size[0]);
    try testing.expectEqual(@as(usize, 3), t.size[1]);
    try testing.expectEqualSlices(f32, data, t.data.ref());
    defer t.data.unref();
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

test "Tensor::ones" {
    const allocator = testing.allocator;
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).ones(&size, allocator);
    defer t.deinit();

    const expected = [_]f32{ 1, 1, 1, 1, 1, 1 };
    try testing.expectEqualSlices(f32, &expected, t.data);
    try testing.expectEqual(@as(usize, 2), t.size[0]);
    try testing.expectEqual(@as(usize, 3), t.size[1]);
}

test "Tensor::transpose" {
    const allocator = testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer t.deinit();

    var t_transposed = try t.transpose(0, 1);
    defer t_transposed.deinit();

    // Make transpose contiguous so that
    try t_transposed.contiguous();

    const expected_data = [_]f32{ 1, 4, 2, 5, 3, 6 };
    const expected_size = [_]usize{ 3, 2 };

    try testing.expectEqualSlices(f32, &expected_data, t_transposed.data);
    try testing.expectEqualSlices(usize, &expected_size, t_transposed.size);
}

test "Tensor::is_contiguous" {
    const allocator = testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer t.deinit();

    try testing.expect(t.is_contiguous());

    var t_transposed = try t.transpose(0, 1);
    defer t_transposed.deinit();

    try testing.expect(!t_transposed.is_contiguous());

    try t_transposed.contiguous();
    try testing.expect(t_transposed.is_contiguous());
}

test "Tensor::contiguous" {
    const allocator = testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer t.deinit();

    // Transpose to make it non-contiguous
    var t_transposed = try t.transpose(0, 1);
    defer t_transposed.deinit();

    // Make it contiguous
    try t_transposed.contiguous();

    const expected_data = [_]f32{ 1, 4, 2, 5, 3, 6 };
    const expected_size = [_]usize{ 3, 2 };

    try testing.expectEqualSlices(f32, &expected_data, t_transposed.data);
    try testing.expectEqualSlices(usize, &expected_size, t_transposed.size);
}

test "Tensor::reshape" {
    const allocator = testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const size = [_]usize{ 2, 3 };

    var t = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer t.deinit();

    // Reshape to 3x2
    const new_size = [_]usize{ 3, 2 };
    var reshaped_t = try t.resize(&new_size);
    defer reshaped_t.deinit();

    const expected_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const expected_size = [_]usize{ 3, 2 };

    try testing.expectEqualSlices(f32, &expected_data, reshaped_t.data);
    try testing.expectEqualSlices(usize, &expected_size, reshaped_t.size);
}

test "Tensor::reshape non-contiguous" {
    const allocator = testing.allocator;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };

    var t = try Tensor(f32).initFromSlice(&.{ 3, 2 }, &data, allocator);
    defer t.deinit();

    var t_transposed = try t.transpose(0, 1);
    defer t_transposed.deinit();

    try testing.expect(!t_transposed.is_contiguous());
    try testing.expectEqualSlices(usize, t_transposed.size, &.{ 2, 3 });

    // Reshape to 3x2
    var reshaped_t = try t_transposed.resize(&.{ 3, 2 });
    defer reshaped_t.deinit();

    const expected_data = [_]f32{ 1, 3, 5, 2, 4, 6 };
    const expected_size = [_]usize{ 3, 2 };

    try testing.expectEqualSlices(f32, &expected_data, reshaped_t.data);
    try testing.expectEqualSlices(usize, &expected_size, reshaped_t.size);
    try testing.expect(reshaped_t.is_contiguous());
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

    var c = try _add(f32, &a, &b);
    defer c.deinit();

    const expected = [_]f32{ 6, 8, 10, 12 };
    try testing.expectEqualSlices(f32, &expected, c.data);

    // Check computation graph
    try testing.expect(c.comp_graph != null);
    try testing.expectEqual(c.comp_graph.?.operation, Op.add);
    try testing.expectEqualSlices(
        *const Tensor(f32),
        c.comp_graph.?.operands.?,
        &.{ &a, &b },
    );
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

    var c = try _matmul(f32, &a, &b);
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

    // Check computation graph
    try testing.expect(c.comp_graph != null);
    try testing.expectEqual(c.comp_graph.?.operation, Op.mul);
    try testing.expectEqualSlices(
        *const Tensor(f32),
        c.comp_graph.?.operands.?,
        &.{ &a, &b },
    );
}

test "mul transpose" {
    const allocator = testing.allocator;

    // Test case for 2x3 * 3x2 matrix multiplication
    const data_a = [_]f32{ 1, 2, 3, 4 }; // 2x2 matrix
    const size_a = [_]usize{ 2, 2 };
    const data_b = [_]f32{ 7, 8, 9, 10 }; // 2x2 matrix
    const size_b = [_]usize{ 2, 2 };

    var a = try Tensor(f32).initFromSlice(&size_a, &data_a, allocator);
    defer a.deinit();
    var b = try Tensor(f32).initFromSlice(&size_b, &data_b, allocator);
    defer b.deinit();

    var b_transpose = try b.transpose(0, 1);
    defer b_transpose.deinit();

    var c = try _matmul(f32, &a, &b_transpose);
    defer c.deinit();

    // [1 2] * [7  9] = [23 29]
    // [3 4]   [8 10]   [53 67]
    const expected = [_]f32{ 23, 29, 53, 67 };

    // Check dimensions
    try testing.expectEqual(@as(usize, 2), c.size[0]);
    try testing.expectEqual(@as(usize, 2), c.size[1]);

    // Check values
    try testing.expectEqualSlices(f32, &expected, c.data);

    // Check computation graph
    try testing.expect(c.comp_graph != null);
    try testing.expectEqual(c.comp_graph.?.operation, Op.mul);
    try testing.expectEqualSlices(
        *const Tensor(f32),
        c.comp_graph.?.operands.?,
        &.{ &a, &b_transpose },
    );
}

test "backward (add)" {
    const allocator = testing.allocator;

    // Create two tensors a and b
    const data_a = [_]f32{ 1, 2, 3, 4 };
    const data_b = [_]f32{ 5, 6, 7, 8 };
    const size = [_]usize{ 2, 2 };

    var a = try Tensor(f32).initFromSlice(&size, &data_a, allocator);
    defer a.deinit();
    var b = try Tensor(f32).initFromSlice(&size, &data_b, allocator);
    defer b.deinit();

    // Perform addition
    var c = try _add(f32, &a, &b);
    defer c.deinit();

    // Perform backward pass
    try _backward(f32, &c);

    // Check gradients
    const expected_grad = [_]f32{ 1, 1, 1, 1 };
    try testing.expectEqualSlices(f32, &expected_grad, a.gradient.?);
    try testing.expectEqualSlices(f32, &expected_grad, b.gradient.?);
}

test "backward (add) #2" {
    const allocator = testing.allocator;

    // Create two tensors a and b
    const data = [_]f32{ 1, 2, 3, 4 };
    const size = [_]usize{ 2, 2 };

    var a = try Tensor(f32).initFromSlice(&size, &data, allocator);
    defer a.deinit();

    // Perform addition
    var c = try _add(f32, &a, &a);
    defer c.deinit();

    // Perform backward pass
    try _backward(f32, &c);

    // Check gradients
    const expected_grad = [_]f32{ 2, 2, 2, 2 };
    try testing.expectEqualSlices(f32, &expected_grad, a.gradient.?);
}

test "backwards nested" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const size = [_]usize{ 2, 2 };

    const data_in0 = [_]f32{ 1, 2, 3, 4 };
    var in0 = try Tensor(f32).initFromSlice(&size, &data_in0, allocator);
    defer in0.deinit();

    const data_in1 = [_]f32{ 5, 6, 7, 8 };
    var in1 = try Tensor(f32).initFromSlice(&size, &data_in1, allocator);
    defer in1.deinit();

    const data_in2 = [_]f32{ 9, 10, 11, 12 };
    var in2 = try Tensor(f32).initFromSlice(&size, &data_in2, allocator);
    defer in2.deinit();

    const data_in3 = [_]f32{ 1, 2, 3, 4 };
    var in3 = try Tensor(f32).initFromSlice(&size, &data_in3, allocator);
    defer in3.deinit();

    // Perform addition
    var out0 = try in0.matmul(&in1);
    defer out0.deinit();
    var out1 = try out0.add(&in2);
    defer out1.deinit();
    var out2 = try out1.matmul(&in3);
    defer out2.deinit();

    // Perform backward pass
    try _backward(f32, &out2);

    // Expected gradients
    const ex0 = [_]f32{ 57, 77, 57, 77 };
    const ex1 = [_]f32{ 12, 28, 18, 42 };
    const ex2 = [_]f32{ 3, 7, 3, 7 };
    const ex3 = [_]f32{ 82, 82, 94, 94 };

    inline for (.{ in0, in1, in2, in3 }, .{ ex0, ex1, ex2, ex3 }) |in, gr| {
        try testing.expectEqualSlices(f32, in.gradient.?, &gr);
    }
}
