const std = @import("std");
const atomic = std.atomic;
const testing = std.testing;

pub fn Arc(comptime T: type) type {
    return struct {
        const Self = @This();

        raw: *T,
        count: atomic.Value(usize),
        dropFn: *const fn (*T) void,

        // Init Arc with initial refcount of 1
        pub fn init(raw: *T, dropFn: *const fn (*T) void) Self {
            return Self{
                .raw = raw,
                .count = atomic.Value(usize).init(1),
                .dropFn = dropFn,
            };
        }

        // Release the initial reference
        pub fn deinit(self: *Self) void {
            self.unref();
        }

        pub fn ref(self: *Self) *T {
            _ = self.count.fetchAdd(1, .monotonic);
            return self.raw;
        }

        pub fn unref(self: *Self) void {
            if (self.count.fetchSub(1, .release) == 1) {
                _ = self.count.load(.acquire);
                (self.dropFn)(self.raw);
            }
        }
    };
}

test "Arc" {
    const allocator = testing.allocator;
    const Dummy = struct {
        const Self = @This();
        data: []u32,

        pub fn init() !Self {
            var data = try allocator.alloc(u32, 10);
            for (0..10) |i| data[i] = @intCast(i);

            return Self{ .data = data };
        }

        pub fn deinit(t: *Self) void {
            allocator.free(t.data);
        }
    };

    // Case 1: Multiple refs
    var dummy = try Dummy.init();

    var dummy_rc = Arc(Dummy).init(&dummy, Dummy.deinit);
    defer dummy_rc.deinit();

    const ref0 = dummy_rc.ref();
    defer dummy_rc.unref();

    const ref1 = dummy_rc.ref();
    defer dummy_rc.unref();

    try testing.expectEqual(ref0.data[0], 0);
    try testing.expectEqual(ref1.data[1], 1);

    // Case 2: No ref
    var dummy_2 = try Dummy.init();

    var dummy_rc_2 = Arc(Dummy).init(&dummy_2, Dummy.deinit);
    defer dummy_rc_2.deinit();
}
