const std = @import("std");
const Allocator = std.mem.Allocator;

/// Tensor over generic T
pub fn Tensor(comptime T: type) Tensor {
    return struct {
        allocator: Allocator,
        data: []T,
        size: []usize,
        stride: []usize,

        inline fn buildStride(size: []const usize, stride: []usize) void {
            // `data` is organized so that values in the innermost layer is
            // adjacent. With every layer on top, we have to multiply by the
            // size of the layer.
            // For example if the shape of the tensor is (3,4,5), this loop
            // will set a stride (1,3,12).
            var running_stride: usize = 1;
            var i: usize = size.len;
            while (i > 0) : (i -= 1) {
                stride[i] = running_stride;
                running_stride *= size[i];
            }
        }

        /// Initialize with owned data
        pub fn initFromOwned(size: []const usize, data: []T, allocator: Allocator) !Tensor {
            const size_heap = try allocator.alloc(T, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(T, size.len);
            buildStride(size, stride_heap);

            return Tensor{
                .allocator = allocator,
                .data = data,
                .size = size_heap,
                .stride = stride_heap,
            };
        }

        /// Initialize with a slice
        pub fn initFromSlice(size: []const usize, data: []const T, allocator: Allocator) !Tensor {
            const data_heap = try allocator.alloc(T, data.len);
            @memcpy(data_heap, data);

            const size_heap = try allocator.alloc(usize, size.len);
            @memcpy(size_heap, size);

            const stride_heap = try allocator.alloc(usize, size.len);
            buildStride(size, stride_heap);

            return Tensor{
                .allocator = allocator,
                .data = data_heap,
                .size = size_heap,
                .stride = stride_heap,
            };
        }

        pub fn deinit(self: Tensor) void {
            self.allocator.free(self.data);
            self.allocator.free(self.size);
            self.allocator.free(self.stride);
        }

        /// Print data
        pub fn print(self: Tensor) void {
            for (self.data) |d| {
                std.debug.print("{} ", .{d});
            }
            std.debug.print("\n", .{});
        }
    };
}
