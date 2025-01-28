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
    var out0 = try in0.mul_(&in1);
    defer out0.deinit();

    std.debug.print("out0: ", .{});
    out0.print();

    var out1 = try out0.add_(&in2);
    defer out1.deinit();

    std.debug.print("out1: ", .{});
    out1.print();

    var out2 = try out1.mul_(&in3);
    defer out2.deinit();

    std.debug.print("out2: ", .{});
    out2.print();

    // Perform backward pass
    try tensor.backward(f32, &out2);

    inline for (.{ in0, in1, in2, in3 }, 0..) |in, i| {
        std.debug.print("Grad in{}: ", .{i});
        printSlice(f32, in.gradient.?);
        std.debug.print("\n", .{});
    }

    inline for (.{ out0, out1, out2 }, 0..) |out_t, i| {
        std.debug.print("Grad out{}: ", .{i});
        printSlice(f32, out_t.gradient.?);
        std.debug.print("\n", .{});
    }
}
