const std = @import("std");
const atomic = std.atomic;
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub fn ArcArray(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        count: *atomic.Value(usize),
        raw: []T,

        // Init Arc with initial refcount of 1
        pub fn init(raw: []T, allocator: Allocator) !Self {
            const count = try allocator.create(atomic.Value(usize));
            count.* = atomic.Value(usize).init(1);

            return Self{
                .allocator = allocator,
                .count = count,
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
                self.allocator.destroy(self.count);
            }
        }

        pub fn clone(self: *Self) Self {
            _ = self.count.fetchAdd(1, .monotonic);
            return Self{
                .allocator = self.allocator,
                .count = self.count,
                .raw = self.raw,
            };
        }
    };
}

test "ArcArray" {
    const allocator = testing.allocator;

    // Case 1: Multiple refs
    var dummy = try allocator.alloc(u32, 10);
    for (0..10) |i| dummy[i] = @intCast(i);

    var dummy_rc = try ArcArray(u32).init(dummy, allocator);
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

    var dummy_rc_2 = try ArcArray(f32).init(dummy2, allocator);
    defer dummy_rc_2.deinit();

    // Case 3: Clone
    try testing.expectEqual(dummy_rc.count.raw, 3);

    var dummy_rc_clone = dummy_rc.clone();
    defer dummy_rc_clone.deinit();

    try testing.expectEqual(dummy_rc.count.raw, 4);
    try testing.expectEqual(dummy_rc_clone.count.raw, 4);

    const ref_clone = dummy_rc_clone.ref();
    defer dummy_rc_clone.unref();

    try testing.expectEqual(ref_clone[0], 0);
    try testing.expectEqual(ref_clone[1], 1);
}
