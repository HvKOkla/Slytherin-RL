module tb_adder;
    logic [7:0] a, b;
    logic [8:0] sum;

    
    adder dut (.a(a), .b(b), .sum(sum));

    initial begin
        $display("--- Test de l'additionneur ---");
        
      
        a = 8'd2; b = 8'd3; #10;
        if (sum == 9'd5) $display("✅ Test 1 OK (2+3=5)");
        else             $display("❌ Test 1 FAIL (attendu 5, recu %0d)", sum);

        
        a = 8'd255; b = 8'd1; #10;
        if (sum == 9'd256) $display("✅ Test 2 OK (255+1=256)");
        else               $display("❌ Test 2 FAIL (attendu 256, recu %0d)", sum);

        $finish;
    end
endmodule