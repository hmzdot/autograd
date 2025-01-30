const std = @import("std");
const atomic = std.atomic;
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub fn ArcArray(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        count: atomic.Value(usize),
        raw: []T,

        // Init Arc with initial refcount of 1
        pub fn init(raw: []T, allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .count = atomic.Value(usize).init(1),
                .raw = raw,
            };
        }

        // Release the initial reference
        pub fn deinit(self: *Self) void {
            self.unref();
        }

        pub fn ref(self: *Self) []T {
            _ = self.count.fetchAdd(1, .monotonic);
            return self.raw;
        }

        pub fn unref(self: *Self) void {
            if (self.count.fetchSub(1, .release) == 1) {
                _ = self.count.load(.acquire);
                self.allocator.free(self.raw);
            }
        }
    };
}

test "ArcArray" {
    const allocator = testing.allocator;

    // Case 1: Multiple refs
    var dummy = try allocator.alloc(u32, 10);
    for (0..10) |i| dummy[i] = @intCast(i);

    var dummy_rc = ArcArray(u32).init(dummy, allocator);
    defer dummy_rc.deinit();

    const ref0 = dummy_rc.ref();
    defer dummy_rc.unref();

    const ref1 = dummy_rc.ref();
    defer dummy_rc.unref();

    try testing.expectEqual(ref0[0], 0);
    try testing.expectEqual(ref1[1], 1);

    // Case 2: No ref
    var dummy2 = try allocator.alloc(f32, 10);
    for (0..10) |i| dummy2[i] = @floatFromInt(i);

    var dummy_rc_2 = ArcArray(f32).init(dummy2, allocator);
    defer dummy_rc_2.deinit();
}
