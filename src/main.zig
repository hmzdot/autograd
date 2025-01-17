const std = @import("std");

const tensor = @import("./tensor.zig");
const Tensor = tensor.Tensor;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = try allocator.alloc(u32, 9);
    for (data, 0..) |*d, i| d.* = @intCast(i);

    const size = [_]usize{ 3, 3 };
    var t = try Tensor(u32).initFromOwned(&size, data, allocator);
    defer t.deinit();

    const t1_size = [_]usize{ 4, 3, 2 };
    const t1_data = .{ 7, 1, 2, 2, 2, 4, 1, 1, 1, 1, 7, 1, 4, 9, 1, 5, 9, 6, 9, 3, 5, 9, 4, 1 };
    var t1 = try Tensor(u32).initFromSlice(&t1_size, &t1_data, allocator);
    defer t1.deinit();

    const t2_size = [_]usize{ 2, 1 };
    const t2_data = .{ 1, 7 };
    var t2 = try Tensor(u32).initFromSlice(&t2_size, &t2_data, allocator);
    defer t2.deinit();

    // tensor([[[14], [16], [30]], [[ 8], [ 8], [14]], [[67], [36], [51]], [[30], [68], [11]]])
    try tensor.mul(u32, &t1, &t2);
}
