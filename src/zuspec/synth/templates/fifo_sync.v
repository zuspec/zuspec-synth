// fifo_sync.v
// Synchronous 2-entry FIFO — Verilog 2005 (IEEE 1364-2005), Yosys compatible.
//
// Parameters:
//   WIDTH  — data width in bits (default 32)
//   DEPTH  — nominal depth annotation; internal storage is fixed at 2 entries
//            (increase if deeper buffering is needed)
//
// Interface:
//   wr_en / din   — write port
//   rd_en / dout  — read port
//   full / empty  — status flags

module fifo_sync #(
  parameter WIDTH = 32,
  parameter DEPTH = 2
) (
  input  wire             clk,
  input  wire             rst_n,
  input  wire             wr_en,
  input  wire [WIDTH-1:0] din,
  output wire [WIDTH-1:0] dout,
  output wire             full,
  output wire             empty,
  input  wire             rd_en
);
  // Fixed 2-entry storage; DEPTH parameter is for documentation only
  reg [WIDTH-1:0] data [0:1];
  reg [1:0]       count;
  reg             wr_ptr;
  reg             rd_ptr;

  assign full  = (count == 2'd2);
  assign empty = (count == 2'd0);
  assign dout  = data[rd_ptr];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      count  <= 2'd0;
      wr_ptr <= 1'b0;
      rd_ptr <= 1'b0;
      data[0] <= {WIDTH{1'b0}};
      data[1] <= {WIDTH{1'b0}};
    end else begin
      if (wr_en && !full && rd_en && !empty) begin
        // Simultaneous read and write
        data[wr_ptr] <= din;
        wr_ptr       <= ~wr_ptr;
        rd_ptr       <= ~rd_ptr;
        // count stays the same
      end else if (wr_en && !full) begin
        data[wr_ptr] <= din;
        wr_ptr       <= ~wr_ptr;
        count        <= count + 2'd1;
      end else if (rd_en && !empty) begin
        rd_ptr <= ~rd_ptr;
        count  <= count - 2'd1;
      end
    end
  end
endmodule
