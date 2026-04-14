// skid_buffer.v
// Skid buffer (2-entry pipeline register with ready/valid handshake).
// Verilog 2005 (IEEE 1364-2005) — Yosys / Verilator compatible.
//
// A skid buffer decouples a producer and consumer so that the consumer
// can de-assert ready without stalling the producer for more than one
// cycle.  This is the standard building block used on critical pipeline
// paths where a pure FIFO would add too many register stages.
//
// Parameters:
//   WIDTH — payload data width in bits (default 32)
//
// Interface:
//   Producer side (upstream):
//     s_valid / s_data — incoming data
//     s_ready         — back-pressure to producer
//   Consumer side (downstream):
//     m_valid / m_data — forwarded data
//     m_ready          — back-pressure from consumer

module skid_buffer #(
  parameter WIDTH = 32
) (
  input  wire             clk,
  input  wire             rst_n,
  // Producer (slave) port
  input  wire             s_valid,
  input  wire [WIDTH-1:0] s_data,
  output wire             s_ready,
  // Consumer (master) port
  output reg              m_valid,
  output reg  [WIDTH-1:0] m_data,
  input  wire             m_ready
);
  // Internal buffer (overflow slot)
  reg             buf_valid;
  reg [WIDTH-1:0] buf_data;

  // Accept new data whenever the overflow slot is empty
  assign s_ready = !buf_valid;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_valid   <= 1'b0;
      m_data    <= {WIDTH{1'b0}};
      buf_valid <= 1'b0;
      buf_data  <= {WIDTH{1'b0}};
    end else begin
      if (m_ready || !m_valid) begin
        // Consumer accepted (or output was empty): shift buffer → output
        if (buf_valid) begin
          m_valid   <= 1'b1;
          m_data    <= buf_data;
          buf_valid <= s_valid && !m_ready;
          buf_data  <= s_data;
        end else begin
          m_valid   <= s_valid;
          m_data    <= s_data;
          buf_valid <= 1'b0;
        end
      end else begin
        // Consumer stalled: capture incoming data into overflow slot
        if (s_valid && s_ready) begin
          buf_valid <= 1'b1;
          buf_data  <= s_data;
        end
      end
    end
  end
endmodule
