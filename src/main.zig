const std = @import("std");

const tensor = @import("./tensor.zig");
const Tensor = tensor.Tensor;

fn printSlice(comptime T: type, slice: []const T) void {
    std.debug.print("[", .{});
    for (slice[0..(slice.len - 1)]) |s| {
        std.debug.print("{},", .{s});
    }
    if (slice.len > 0) {
        std.debug.print("{}", .{slice[slice.len - 1]});
    }
    std.debug.print("]", .{});
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var t = try Tensor(u32).initFromSlice(
        &.{ 6, 2, 1 },
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        allocator,
    );

    std.debug.print("size: ", .{});
    printSlice(usize, t.size);
    std.debug.print("\n", .{});

    std.debug.print("stride: ", .{});
    printSlice(usize, t.stride);
    std.debug.print("\n", .{});

    try t.contiguous();

    t.print();
}
