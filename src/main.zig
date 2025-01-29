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
    const ator = arena.allocator();

    var in0 = try Tensor(f32).initFromSlice(
        &.{ 2, 2 },
        &.{ 1, 2, 3, 4 },
        ator,
    );
    defer in0.deinit();
    var in1 = try Tensor(f32).initFromSlice(
        &.{ 2, 2 },
        &.{ 5, 6, 7, 8 },
        ator,
    );
    defer in1.deinit();
    var in2 = try Tensor(f32).initFromSlice(
        &.{ 2, 2 },
        &.{ 9, 10, 11, 12 },
        ator,
    );
    defer in2.deinit();

    // out = in0 * in1 + in2
    var out0 = try in0.matmul(&in1);
    defer out0.deinit();
    var out1 = try out0.add(&in2);
    defer out1.deinit();

    // Perform backward pass and calculate gradients
    try out1.backward();

    inline for (.{ in0, in1, in2 }, 0..) |in, i| {
        std.debug.print("Grad in{}: ", .{i});
        printSlice(f32, in.gradient.?);
        std.debug.print("\n", .{});
    }
}
