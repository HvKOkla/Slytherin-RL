module mac (
    input  logic               clk,
    input  logic               rst_n,    
    input  logic signed [7:0]  a,         
    input  logic signed [7:0]  b,         
    input  logic               valid,  
    output logic signed [31:0] acc        
);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc <= 32'd0; 
        end else if (valid) begin
            acc <= acc + (a * b);
        end
    end
endmodule