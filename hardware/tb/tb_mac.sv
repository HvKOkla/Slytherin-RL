module tb_mac;
    logic clk, rst_n, valid;
    logic signed [7:0] a, b;
    logic signed [31:0] acc;

    // Instanciation du module
    mac dut (
        .clk(clk), .rst_n(rst_n), 
        .a(a), .b(b), .valid(valid), 
        .acc(acc)
    );

    // Génération de l'horloge (période de 10 unités)
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $display("--- Test du MAC ---");
        
        // 1. Reset
        rst_n = 0; valid = 0; a = 0; b = 0;
        #10 rst_n = 1; // On enlève le reset

        // 2. Premier calcul : 2 * 3 = 6
        a = 8'd2; b = 8'd3; valid = 1;
        #10; // On attend un front d'horloge
        if (acc == 32'd6) $display("Test 1 OK: 0 + (2*3) = %0d", acc);
        else              $display("Test 1 FAIL: %0d", acc);

        // 3. Deuxième calcul (accumulation) : 6 + (4 * 5) = 26
        a = 8'd4; b = 8'd5; valid = 1;
        #10;
        if (acc == 32'd26) $display("Test 2 OK: 6 + (4*5) = %0d", acc);
        else               $display("Test 2 FAIL: %0d", acc);

        // 4. Test avec des nombres négatifs : 26 + (-3 * 2) = 20
        a = -8'sd3; b = 8'sd2; valid = 1;
        #10;
        if (acc == 32'd20) $display("Test 3 OK: 26 + (-3*2) = %0d", acc);
        else               $display("Test 3 FAIL: %0d", acc);

        
        valid = 0;
        #10;
        if (acc == 32'd20) $display("Test 4 OK: Memoire conservee = %0d", acc);
        else               $display("Test 4 FAIL: %0d", acc);

        $finish;
    end
endmodule