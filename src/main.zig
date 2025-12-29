const std = @import("std");
const ml = @import("ml.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var weights = try ml.Matrix.init(allocator, 3, 2);
    defer weights.deinit();
    weights.fillRandom(42);

    var bias = try ml.Matrix.init(allocator, 1, 2);
    defer bias.deinit();
    bias.set(0, 0, 0.1);
    bias.set(0, 1, 0.1);

    var input = try ml.Matrix.init(allocator, 1, 3);
    defer input.deinit();
    input.set(0, 0, 1.0);
    input.set(0, 1, 0.5);
    input.set(0, 2, -1.0);

    var output = try ml.Matrix.dot(input, weights, allocator);
    defer output.deinit();
    
    output.add(bias);
    
    ml.applySigmoid(&output);

    input.print("Input");
    weights.print("Weights");
    output.print("Final Output (after Sigmoid)");
}
