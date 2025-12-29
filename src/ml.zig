const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return Matrix{
            .rows = rows,
            .cols = cols,
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: Matrix) void {
        self.allocator.free(self.data);
    }

    pub fn at(self: Matrix, r: usize, c: usize) f32 {
        return self.data[r * self.cols + c];
    }

    pub fn set(self: *Matrix, r: usize, c: usize, val: f32) void {
        self.data[r * self.cols + c] = val;
    }

    pub fn dot(a: Matrix, b: Matrix, allocator: Allocator) !Matrix {
        std.debug.assert(a.cols == b.rows);
        var res = try Matrix.init(allocator, a.rows, b.cols);
        for (0..a.rows) |i| {
            for (0..b.cols) |j| {
                var sum: f32 = 0;
                for (0..a.cols) |k| {
                    sum += a.at(i, k) * b.at(k, j);
                }
                res.set(i, j, sum);
            }
        }
        return res;
    }

    pub fn add(self: *Matrix, other: Matrix) void {
        std.debug.assert(self.rows == other.rows);
        std.debug.assert(self.cols == other.cols);
        for (0..self.data.len) |i| {
            self.data[i] += other.data[i];
        }
    }

    pub fn fillRandom(self: *Matrix, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (self.data) |*val| {
            val.* = (random.float(f32) * 2.0) - 1.0;
        }
    }

    pub fn print(self: Matrix, label: []const u8) void {
        std.debug.print("{s} ({d}x{d}):\n", .{ label, self.rows, self.cols });
        for (0..self.rows) |i| {
            std.debug.print("  ", .{});
            for (0..self.cols) |j| {
                std.debug.print("{d: >8.4} ", .{self.at(i, j)});
            }
            std.debug.print("\n", .{});
        }
    }
};


pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub fn applySigmoid(m: *Matrix) void {
    for (m.data) |*val| {
        val.* = sigmoid(val.*);
    }
}
