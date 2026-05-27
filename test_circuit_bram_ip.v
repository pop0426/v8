module test_circuit #(
    parameter MEM_DW = 32,
    parameter MEM_AW = 10
)(
    //global
    input  clk,
    input  rstn,
    output reg sys_rstn,
    //data mem
    output [MEM_AW-1:0] mem_addr,
    output [MEM_DW-1:0] mem_wdata,
    input  [MEM_DW-1:0] mem_rdata,
    output              mem_we,
    //cpu
    input  [MEM_AW-1:0] cpu_addr,
    input  [MEM_DW-1:0] cpu_wdata,
    output [MEM_DW-1:0] cpu_rdata,
    input               cpu_we,
    input               system_done,
    //user control
    input               start_bt,
    input               mode,    // 0: run / testcase result mode, 1: score display mode
    input  [3:0]        tc,      // 0~9: CNN, 10~14: CPU subtests, 15: run all / summary
    // Display output
    output reg [7-1:0] seven_seg,
    output reg [3:0] anode,
    output reg all_done
);

localparam TEST_NUM = 10;
localparam INIT_WORDS = 1024;
localparam MEM_WORDS = 1024;
localparam RESERVED_START = 960;
localparam RESERVED_END = 1023;
localparam GOLDEN_MAX_WORDS = 196;
localparam CPU_SUBTEST_NUM = 5;
localparam [11:0] CPU_GOLDEN_BASE = TEST_NUM * GOLDEN_MAX_WORDS;
localparam CLK_PERIOD_NS = 10;
localparam [31:0] RUN_TIMEOUT_CYCLES = 32'd100000000;
localparam [19:0] DEBOUNCE_MAX = 20'd999999;

// S_IDLE: wait for start button while holding CPU/CNN reset.
// S_CLEAR_MEM: write zero to data_mem[0:1023].
// S_LOAD_INIT_REQ: issue synchronous ROM read for selected init word.
// S_LOAD_INIT_WRITE: write selected init ROM word to data_mem[0:1023].
// S_WRITE_START_REQ: issue synchronous ROM read for init word at address 11.
// S_WRITE_START_WRITE: force data_mem[11][0] to 1.
// S_RELEASE_SYS_RESET: release CPU/CNN reset for one cycle before RUN.
// S_RUN: CPU owns Port A; test circuit monitors finish write.
// S_CHECK_PREP: take back Port A and reset CPU/CNN.
// S_READ_ADDR13_REQ: issue synchronous read of data_mem[13].
// S_READ_ADDR13_CAPTURE: capture data_mem[13] and derived count values.
// S_READ_ADDR13_EVAL: read finish bit and actual output start field from captured data.
// S_READ_RESERVED_REQ: issue synchronous read of CPU test region Data Memory[960:1023]
//                      and CPU golden block golden[196*10 + 0 : 196*10 + 63].
// S_READ_RESERVED_CAPTURE: capture BRAM/ROM data for CPU test region compare.
// S_READ_RESERVED_EVAL: compare CPU test region with CPU golden block and update
//                      five CPU subtest fail flags independently.
// S_READ_OFMAP_REQ: issue synchronous read of output feature map word and golden ROM word.
// S_READ_OFMAP_CAPTURE: capture BRAM/ROM data for output feature map compare.
// S_READ_OFMAP_EVAL: compare output feature map word with golden ROM.
// S_UPDATE_SCORE: update logical testcase pass/fail bits.
// S_NEXT_TEST: advance to next testcase or finish.
// S_DONE: show final result and wait in inactive state.
localparam [4:0] S_IDLE               = 5'd0;
localparam [4:0] S_CLEAR_MEM          = 5'd1;
localparam [4:0] S_LOAD_INIT_REQ      = 5'd2;
localparam [4:0] S_LOAD_INIT_WRITE    = 5'd3;
localparam [4:0] S_WRITE_START_REQ    = 5'd4;
localparam [4:0] S_WRITE_START_WRITE  = 5'd5;
localparam [4:0] S_RELEASE_SYS_RESET  = 5'd6;
localparam [4:0] S_RUN                = 5'd7;
localparam [4:0] S_CHECK_PREP         = 5'd8;
localparam [4:0] S_READ_ADDR13_REQ    = 5'd9;
localparam [4:0] S_READ_RESERVED_REQ  = 5'd11;
localparam [4:0] S_READ_OFMAP_REQ     = 5'd13;
localparam [4:0] S_UPDATE_SCORE       = 5'd15;
localparam [4:0] S_NEXT_TEST          = 5'd16;
localparam [4:0] S_DONE               = 5'd17;
localparam [4:0] S_READ_ADDR13_CAPTURE   = 5'd18;
localparam [4:0] S_READ_ADDR13_EVAL      = 5'd19;
localparam [4:0] S_READ_RESERVED_CAPTURE = 5'd20;
localparam [4:0] S_READ_RESERVED_EVAL    = 5'd21;
localparam [4:0] S_READ_OFMAP_CAPTURE    = 5'd22;
localparam [4:0] S_READ_OFMAP_EVAL       = 5'd23;

reg [4:0] state, nstate;

// Internal registered flag for testbench simulation stop condition.
// This signal is not exported to the top-level port list.

reg test_mode;
reg [MEM_AW-1:0] tc_addr;
reg [MEM_DW-1:0] tc_wdata;
reg tc_we;

reg [MEM_AW-1:0] addr_counter;
reg [7:0] ofmap_idx;
reg [3:0] test_id;
reg [3:0] selected_tc;
reg testcase_fail;  // CNN testcase fail flag only.
reg cpu_addi_sw_fail, cpu_lw_fail, cpu_add_sub_fail, cpu_beq_fail, cpu_blt_fail;
reg [14:0] result_valid;
reg [14:0] result_pass;
wire [3:0] cpu_pass_score;
wire [3:0] cnn_pass_count;
wire [3:0] pass_count, fail_count;
wire [7:0] weighted_score_decimal; // integer score, 0~100
reg [11:0] tc_bcd;
reg [3:0] status_digit;
reg [15:0] total_pass_bcd;
reg [15:0] score_display_bcd;
reg [31:0] cycle_counter;
reg [31:0] captured_addr13;
reg [MEM_AW-1:0] captured_ofmap_start;
reg [31:0] captured_addr13_q;
reg [MEM_AW-1:0] captured_ofmap_start_q;

wire [MEM_DW-1:0] init_word, golden_word;
reg [MEM_AW-1:0] init_lookup_addr, ofmap_addr_offset;
reg [5:0] feature_size;
reg [7:0] ofmap_word_count;
reg [7:0] ofmap_word_count_q;
reg [11:0] final_ofmap_size_ext, ofmap_elem_count_ext;
wire finish_event, run_timeout_event;

wire [13:0] init_rom_addr;
wire [13:0] init_rom_base;
wire [13:0] init_test_id_ext;
wire [13:0] init_lookup_addr_ext;
wire [11:0] golden_rom_addr;
wire [11:0] golden_rom_base;
wire [11:0] golden_test_id_ext;
wire [11:0] golden_idx_ext;
wire [11:0] cpu_golden_idx_ext;

reg start_sync0, start_sync1, start_stable, start_stable_d;
reg [19:0] start_db_cnt;
wire start_bt_pulse;

reg [15:0] scan_counter;
wire [1:0] scan_sel;
reg [15:0] display_value_q;
reg [1:0] scan_sel_q;
reg [3:0] display_nibble_q;
reg [3:0] anode_q;
reg [7-1:0] seven_seg_q;
reg [MEM_DW-1:0] mem_rdata_q;
reg [MEM_DW-1:0] golden_word_q;
reg [MEM_AW-1:0] addr_counter_q;
reg [7:0] ofmap_idx_q;

assign mem_addr = test_mode ? tc_addr : cpu_addr;
assign mem_wdata = test_mode ? tc_wdata : cpu_wdata;
assign mem_we = test_mode ? tc_we : cpu_we;
assign cpu_rdata = test_mode ? {MEM_DW{1'b0}} : mem_rdata;

assign finish_event = system_done || (cpu_we && (cpu_addr == 10'd13) && cpu_wdata[0]);
assign run_timeout_event = (cycle_counter == RUN_TIMEOUT_CYCLES);
assign start_bt_pulse = start_stable && !start_stable_d;
assign scan_sel = scan_counter[15:14];

assign init_test_id_ext = {10'd0, test_id};
assign init_lookup_addr_ext = {4'd0, init_lookup_addr};
assign init_rom_base = (init_test_id_ext << 10);
assign init_rom_addr = init_rom_base + init_lookup_addr_ext;

assign golden_test_id_ext = {8'd0, test_id};
assign golden_idx_ext = {4'd0, ofmap_idx};
assign cpu_golden_idx_ext = {6'd0, addr_counter[5:0]};
assign golden_rom_base = (golden_test_id_ext << 7) + (golden_test_id_ext << 6) + (golden_test_id_ext << 2);
assign golden_rom_addr = ((state == S_READ_RESERVED_REQ) ||
                          (state == S_READ_RESERVED_CAPTURE) ||
                          (state == S_READ_RESERVED_EVAL)) ?
                         (CPU_GOLDEN_BASE + cpu_golden_idx_ext) :
                         (golden_rom_base + golden_idx_ext);

assign cnn_pass_count = {3'd0, (result_valid[0] && result_pass[0])} +
                        {3'd0, (result_valid[1] && result_pass[1])} +
                        {3'd0, (result_valid[2] && result_pass[2])} +
                        {3'd0, (result_valid[3] && result_pass[3])} +
                        {3'd0, (result_valid[4] && result_pass[4])} +
                        {3'd0, (result_valid[5] && result_pass[5])} +
                        {3'd0, (result_valid[6] && result_pass[6])} +
                        {3'd0, (result_valid[7] && result_pass[7])} +
                        {3'd0, (result_valid[8] && result_pass[8])} +
                        {3'd0, (result_valid[9] && result_pass[9])};
assign cpu_pass_score = {3'd0, (result_valid[10] && result_pass[10])} +
                        {3'd0, (result_valid[11] && result_pass[11])} +
                        {3'd0, (result_valid[12] && result_pass[12])} +
                        {3'd0, (result_valid[13] && result_pass[13])} +
                        {3'd0, (result_valid[14] && result_pass[14])};
assign pass_count = cnn_pass_count + cpu_pass_score;
assign fail_count = {3'd0, (result_valid[0]  && !result_pass[0])}  +
                    {3'd0, (result_valid[1]  && !result_pass[1])}  +
                    {3'd0, (result_valid[2]  && !result_pass[2])}  +
                    {3'd0, (result_valid[3]  && !result_pass[3])}  +
                    {3'd0, (result_valid[4]  && !result_pass[4])}  +
                    {3'd0, (result_valid[5]  && !result_pass[5])}  +
                    {3'd0, (result_valid[6]  && !result_pass[6])}  +
                    {3'd0, (result_valid[7]  && !result_pass[7])}  +
                    {3'd0, (result_valid[8]  && !result_pass[8])}  +
                    {3'd0, (result_valid[9]  && !result_pass[9])}  +
                    {3'd0, (result_valid[10] && !result_pass[10])} +
                    {3'd0, (result_valid[11] && !result_pass[11])} +
                    {3'd0, (result_valid[12] && !result_pass[12])} +
                    {3'd0, (result_valid[13] && !result_pass[13])} +
                    {3'd0, (result_valid[14] && !result_pass[14])};

// Weighted grading: original 20% rubric scaled to 100 points.
// CNN 10% -> 50 points: each of 10 CNN testcases = 5 points.
// CPU 10% -> 50 points: five CPU subtests, each 2% -> 10 points.
//   addi/sw = 10, lw = 10, add/sub = 10, beq = 10, blt = 10.
// No divider is used here. The constants 5 and 10 are implemented by shifts/adds.
assign weighted_score_decimal = ({4'd0, cnn_pass_count} << 2) + {4'd0, cnn_pass_count} +
                                ({4'd0, cpu_pass_score} << 3) + ({4'd0, cpu_pass_score} << 1);

// Split read/compare into capture/eval states for Fmax improvement
always @(*) begin
    case (state)
        S_IDLE: nstate = (mode == 1'b0 && start_bt_pulse) ? S_CLEAR_MEM : S_IDLE;
        S_CLEAR_MEM: nstate = (addr_counter == 10'd1023) ? S_LOAD_INIT_REQ : S_CLEAR_MEM;
        S_LOAD_INIT_REQ: nstate = S_LOAD_INIT_WRITE;
        S_LOAD_INIT_WRITE: nstate = (addr_counter == 10'd1023) ? S_WRITE_START_REQ : S_LOAD_INIT_REQ;
        S_WRITE_START_REQ: nstate = S_WRITE_START_WRITE;
        S_WRITE_START_WRITE: nstate = S_RELEASE_SYS_RESET;
        S_RELEASE_SYS_RESET: nstate = S_RUN;
        S_RUN: nstate = (finish_event || run_timeout_event) ? S_CHECK_PREP : S_RUN;
        S_CHECK_PREP: nstate = S_READ_ADDR13_REQ;
        S_READ_ADDR13_REQ: nstate = S_READ_ADDR13_CAPTURE;
        S_READ_ADDR13_CAPTURE: nstate = S_READ_ADDR13_EVAL;
        S_READ_ADDR13_EVAL: nstate = S_READ_RESERVED_REQ;
        S_READ_RESERVED_REQ: nstate = S_READ_RESERVED_CAPTURE;
        S_READ_RESERVED_CAPTURE: nstate = S_READ_RESERVED_EVAL;
        S_READ_RESERVED_EVAL: begin
            if (addr_counter_q == 10'd1023) begin
                nstate = ((selected_tc >= 4'd10) && (selected_tc <= 4'd14)) ? S_UPDATE_SCORE : S_READ_OFMAP_REQ;
            end
            else begin
                nstate = S_READ_RESERVED_REQ;
            end
        end
        S_READ_OFMAP_REQ: nstate = S_READ_OFMAP_CAPTURE;
        S_READ_OFMAP_CAPTURE: nstate = S_READ_OFMAP_EVAL;
        S_READ_OFMAP_EVAL: nstate = (ofmap_idx_q == (ofmap_word_count_q - 8'd1)) ? S_UPDATE_SCORE : S_READ_OFMAP_REQ;
        S_UPDATE_SCORE: nstate = S_NEXT_TEST;
        S_NEXT_TEST: nstate = ((selected_tc == 4'd15) && (test_id != 4'd9)) ? S_CLEAR_MEM : S_DONE;
        S_DONE: nstate = (mode == 1'b0 && start_bt_pulse) ? S_CLEAR_MEM : S_DONE;
        default: nstate = S_IDLE;
    endcase
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= S_IDLE;
        all_done <= 1'b0;
        sys_rstn <= 1'b0;
        test_mode <= 1'b1;
        addr_counter <= 10'd0;
        ofmap_idx <= 8'd0;
        test_id <= 4'd0;
        selected_tc <= 4'd0;
        testcase_fail <= 1'b0;
        cpu_addi_sw_fail <= 1'b0;
        cpu_lw_fail <= 1'b0;
        cpu_add_sub_fail <= 1'b0;
        cpu_beq_fail <= 1'b0;
        cpu_blt_fail <= 1'b0;
        result_valid <= 15'd0;
        result_pass <= 15'd0;
        cycle_counter <= 32'd0;
        captured_addr13 <= 32'd0;
        captured_ofmap_start <= 10'd0;
        captured_addr13_q <= 32'd0;
        captured_ofmap_start_q <= 10'd0;
        ofmap_word_count_q <= 8'd0;
        mem_rdata_q <= 32'd0;
        golden_word_q <= 32'd0;
        addr_counter_q <= 10'd0;
        ofmap_idx_q <= 8'd0;
    end
    else begin
        state <= nstate;
        case (state)
            S_IDLE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (mode == 1'b0 && start_bt_pulse) begin
                    all_done <= 1'b0;
                    addr_counter <= 10'd0;
                    ofmap_idx <= 8'd0;
                    selected_tc <= tc;
                    test_id <= (tc <= 4'd9) ? tc : 4'd0;
                    testcase_fail <= 1'b0;
                    cpu_addi_sw_fail <= 1'b0;
                    cpu_lw_fail <= 1'b0;
                    cpu_add_sub_fail <= 1'b0;
                    cpu_beq_fail <= 1'b0;
                    cpu_blt_fail <= 1'b0;
                    if (tc == 4'd15) begin
                        result_valid <= 15'd0;
                        result_pass <= 15'd0;
                    end
                    else begin
                        result_valid[tc] <= 1'b0;
                        result_pass[tc] <= 1'b0;
                    end
                    cycle_counter <= 32'd0;
                    captured_addr13 <= 32'd0;
                    captured_ofmap_start <= 10'd0;
                    captured_addr13_q <= 32'd0;
                    captured_ofmap_start_q <= 10'd0;
                    ofmap_word_count_q <= 8'd0;
                    mem_rdata_q <= 32'd0;
                    golden_word_q <= 32'd0;
                    addr_counter_q <= 10'd0;
                    ofmap_idx_q <= 8'd0;
                end
            end
            S_CLEAR_MEM: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (addr_counter == 10'd1023) begin
                    addr_counter <= 10'd0;
                end
                else begin
                    addr_counter <= addr_counter + 10'd1;
                end
            end
            S_LOAD_INIT_REQ: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
            end
            S_LOAD_INIT_WRITE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (addr_counter == 10'd1023) begin
                    addr_counter <= 10'd0;
                end
                else begin
                    addr_counter <= addr_counter + 10'd1;
                end
            end
            S_WRITE_START_REQ: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
            end
            S_WRITE_START_WRITE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                addr_counter <= 10'd0;
                ofmap_idx <= 8'd0;
                testcase_fail <= 1'b0;
                cycle_counter <= 32'd0;
                captured_addr13 <= 32'd0;
                captured_ofmap_start <= 10'd0;
                captured_addr13_q <= 32'd0;
                captured_ofmap_start_q <= 10'd0;
                ofmap_word_count_q <= 8'd0;
                mem_rdata_q <= 32'd0;
                golden_word_q <= 32'd0;
                addr_counter_q <= 10'd0;
                ofmap_idx_q <= 8'd0;
            end
            S_RELEASE_SYS_RESET: begin
                sys_rstn <= 1'b1;
                test_mode <= 1'b0;
            end
            S_RUN: begin
                sys_rstn <= 1'b1;
                test_mode <= 1'b0;
                if (finish_event) begin
                    captured_addr13 <= cpu_wdata;
                    captured_ofmap_start <= cpu_wdata[10:1];
                end
                else begin
                    if (run_timeout_event) begin
                        captured_addr13 <= 32'd0;
                        captured_ofmap_start <= 10'd0;
                        testcase_fail <= 1'b1;
                        cpu_addi_sw_fail <= 1'b1;
                        cpu_lw_fail <= 1'b1;
                        cpu_add_sub_fail <= 1'b1;
                        cpu_beq_fail <= 1'b1;
                        cpu_blt_fail <= 1'b1;
                    end
                    else begin
                        cycle_counter <= cycle_counter + 32'd1;
                    end
                end
            end
            S_CHECK_PREP: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                addr_counter <= 10'd960;
                ofmap_idx <= 8'd0;
            end
            S_READ_ADDR13_REQ: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
            end
            S_READ_ADDR13_CAPTURE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                captured_addr13_q <= mem_rdata;
                captured_ofmap_start_q <= mem_rdata[10:1];
                ofmap_word_count_q <= ofmap_word_count;
            end
            S_READ_ADDR13_EVAL: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                captured_addr13 <= captured_addr13_q;
                captured_ofmap_start <= captured_ofmap_start_q;
                if ((captured_addr13_q[0] != 1'b1) ||
                    (({1'b0, captured_ofmap_start_q} + {3'd0, ofmap_word_count_q}) > 11'd960)) begin
                    testcase_fail <= 1'b1;
                    cpu_addi_sw_fail <= 1'b1;
                    cpu_lw_fail <= 1'b1;
                    cpu_add_sub_fail <= 1'b1;
                    cpu_beq_fail <= 1'b1;
                    cpu_blt_fail <= 1'b1;
                end
            end
            S_READ_RESERVED_REQ: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
            end
            S_READ_RESERVED_CAPTURE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                // Capture BRAM/ROM outputs before compare to shorten timing path
                mem_rdata_q <= mem_rdata;
                golden_word_q <= golden_word;
                addr_counter_q <= addr_counter;
            end
            S_READ_RESERVED_EVAL: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (mem_rdata_q != golden_word_q) begin
                    case (addr_counter_q)
                        10'd965: cpu_addi_sw_fail <= 1'b1;
                        10'd966: cpu_lw_fail <= 1'b1;
                        10'd967, 10'd968: cpu_add_sub_fail <= 1'b1;
                        10'd969, 10'd970: cpu_beq_fail <= 1'b1;
                        10'd971, 10'd972: cpu_blt_fail <= 1'b1;
                        default: begin
                            // Inputs [960:964] and unused outputs [973:1023]
                            // should remain equal to the CPU golden block. Since
                            // there is no separate score item for illegal writes,
                            // any mismatch here invalidates all CPU subtests.
                            cpu_addi_sw_fail <= 1'b1;
                            cpu_lw_fail <= 1'b1;
                            cpu_add_sub_fail <= 1'b1;
                            cpu_beq_fail <= 1'b1;
                            cpu_blt_fail <= 1'b1;
                        end
                    endcase
                end
                if (addr_counter_q == 10'd1023) begin
                    addr_counter <= 10'd0;
                end
                else begin
                    addr_counter <= addr_counter_q + 10'd1;
                end
            end
            S_READ_OFMAP_REQ: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
            end
            S_READ_OFMAP_CAPTURE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                // Capture BRAM/ROM outputs before compare to shorten timing path
                mem_rdata_q <= mem_rdata;
                golden_word_q <= golden_word;
                ofmap_idx_q <= ofmap_idx;
            end
            S_READ_OFMAP_EVAL: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (mem_rdata_q != golden_word_q) begin
                    testcase_fail <= 1'b1;
                end
                if (ofmap_idx_q == (ofmap_word_count_q - 8'd1)) begin
                    ofmap_idx <= 8'd0;
                end
                else begin
                    ofmap_idx <= ofmap_idx_q + 8'd1;
                end
            end
            S_UPDATE_SCORE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                case (selected_tc)
                    4'd0, 4'd1, 4'd2, 4'd3, 4'd4,
                    4'd5, 4'd6, 4'd7, 4'd8, 4'd9: begin
                        result_valid[selected_tc] <= 1'b1;
                        result_pass[selected_tc] <= !testcase_fail;
                    end
                    4'd10: begin
                        result_valid[10] <= 1'b1;
                        result_pass[10] <= !cpu_addi_sw_fail;
                    end
                    4'd11: begin
                        result_valid[11] <= 1'b1;
                        result_pass[11] <= !cpu_lw_fail;
                    end
                    4'd12: begin
                        result_valid[12] <= 1'b1;
                        result_pass[12] <= !cpu_add_sub_fail;
                    end
                    4'd13: begin
                        result_valid[13] <= 1'b1;
                        result_pass[13] <= !cpu_beq_fail;
                    end
                    4'd14: begin
                        result_valid[14] <= 1'b1;
                        result_pass[14] <= !cpu_blt_fail;
                    end
                    default: begin
                        result_valid[test_id] <= 1'b1;
                        result_pass[test_id] <= !testcase_fail;
                        if (test_id == 4'd9) begin
                            result_valid[10] <= 1'b1;
                            result_pass[10] <= !cpu_addi_sw_fail;
                            result_valid[11] <= 1'b1;
                            result_pass[11] <= !cpu_lw_fail;
                            result_valid[12] <= 1'b1;
                            result_pass[12] <= !cpu_add_sub_fail;
                            result_valid[13] <= 1'b1;
                            result_pass[13] <= !cpu_beq_fail;
                            result_valid[14] <= 1'b1;
                            result_pass[14] <= !cpu_blt_fail;
                        end
                    end
                endcase
            end
            S_NEXT_TEST: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                addr_counter <= 10'd0;
                ofmap_idx <= 8'd0;
                testcase_fail <= 1'b0;
                cycle_counter <= 32'd0;
                captured_addr13 <= 32'd0;
                captured_ofmap_start <= 10'd0;
                captured_addr13_q <= 32'd0;
                captured_ofmap_start_q <= 10'd0;
                ofmap_word_count_q <= 8'd0;
                mem_rdata_q <= 32'd0;
                golden_word_q <= 32'd0;
                addr_counter_q <= 10'd0;
                ofmap_idx_q <= 8'd0;
                if ((selected_tc == 4'd15) && (test_id != 4'd9)) begin
                    test_id <= test_id + 4'd1;
                end
            end
            S_DONE: begin
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                if (mode == 1'b0 && start_bt_pulse) begin
                    all_done <= 1'b0;
                    addr_counter <= 10'd0;
                    ofmap_idx <= 8'd0;
                    selected_tc <= tc;
                    test_id <= (tc <= 4'd9) ? tc : 4'd0;
                    testcase_fail <= 1'b0;
                    cpu_addi_sw_fail <= 1'b0;
                    cpu_lw_fail <= 1'b0;
                    cpu_add_sub_fail <= 1'b0;
                    cpu_beq_fail <= 1'b0;
                    cpu_blt_fail <= 1'b0;
                    if (tc == 4'd15) begin
                        result_valid <= 15'd0;
                        result_pass <= 15'd0;
                    end
                    else begin
                        result_valid[tc] <= 1'b0;
                        result_pass[tc] <= 1'b0;
                    end
                    cycle_counter <= 32'd0;
                    captured_addr13 <= 32'd0;
                    captured_ofmap_start <= 10'd0;
                    captured_addr13_q <= 32'd0;
                    captured_ofmap_start_q <= 10'd0;
                    ofmap_word_count_q <= 8'd0;
                    mem_rdata_q <= 32'd0;
                    golden_word_q <= 32'd0;
                    addr_counter_q <= 10'd0;
                    ofmap_idx_q <= 8'd0;
                end
                else begin
                    // Assert one cycle after the run-all flow reaches S_DONE.
                    // This means all 15 result_valid bits have already been updated.
                    all_done <= (selected_tc == 4'd15) && (result_valid == 15'h7FFF);
                end
            end
            default: begin
                all_done <= 1'b0;
                sys_rstn <= 1'b0;
                test_mode <= 1'b1;
                addr_counter <= 10'd0;
                ofmap_idx <= 8'd0;
                test_id <= 4'd0;
                selected_tc <= 4'd0;
                testcase_fail <= 1'b0;
                cpu_addi_sw_fail <= 1'b0;
                cpu_lw_fail <= 1'b0;
                cpu_add_sub_fail <= 1'b0;
                cpu_beq_fail <= 1'b0;
                cpu_blt_fail <= 1'b0;
                cycle_counter <= 32'd0;
                captured_addr13 <= 32'd0;
                captured_ofmap_start <= 10'd0;
                captured_addr13_q <= 32'd0;
                captured_ofmap_start_q <= 10'd0;
                ofmap_word_count_q <= 8'd0;
                mem_rdata_q <= 32'd0;
                golden_word_q <= 32'd0;
                addr_counter_q <= 10'd0;
                ofmap_idx_q <= 8'd0;
            end
        endcase
    end
end

always @(*) begin
    tc_addr = {MEM_AW{1'b0}};
    tc_wdata = {MEM_DW{1'b0}};
    tc_we = 1'b0;
    case (state)
        S_CLEAR_MEM: begin
            tc_addr = addr_counter;
            tc_wdata = 32'h00000000;
            tc_we = 1'b1;
        end
        S_LOAD_INIT_REQ: begin
            tc_addr = addr_counter;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_LOAD_INIT_WRITE: begin
            tc_addr = addr_counter;
            tc_wdata = init_word;
            tc_we = 1'b1;
        end
        S_WRITE_START_REQ: begin
            tc_addr = 10'd11;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_WRITE_START_WRITE: begin
            tc_addr = 10'd11;
            tc_wdata = {31'd0, 1'b1};
            tc_we = 1'b1;
        end
        S_READ_ADDR13_REQ: begin
            tc_addr = 10'd13;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_ADDR13_CAPTURE: begin
            tc_addr = 10'd13;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_ADDR13_EVAL: begin
            tc_addr = 10'd13;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_RESERVED_REQ: begin
            tc_addr = addr_counter;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_RESERVED_CAPTURE: begin
            tc_addr = addr_counter;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_RESERVED_EVAL: begin
            tc_addr = addr_counter;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_OFMAP_REQ: begin
            tc_addr = captured_ofmap_start + ofmap_addr_offset;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_OFMAP_CAPTURE: begin
            tc_addr = captured_ofmap_start + ofmap_addr_offset;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        S_READ_OFMAP_EVAL: begin
            tc_addr = captured_ofmap_start + ofmap_addr_offset;
            tc_wdata = 32'h00000000;
            tc_we = 1'b0;
        end
        default: begin
            tc_addr = {MEM_AW{1'b0}};
            tc_wdata = {MEM_DW{1'b0}};
            tc_we = 1'b0;
        end
    endcase
end

always @(*) begin
    feature_size = 6'd10;
    case (test_id)
        4'd0: feature_size = 6'd10;
        4'd1: feature_size = 6'd12;
        4'd2: feature_size = 6'd14;
        4'd3: feature_size = 6'd16;
        4'd4: feature_size = 6'd18;
        4'd5: feature_size = 6'd20;
        4'd6: feature_size = 6'd24;
        4'd7: feature_size = 6'd28;
        4'd8: feature_size = 6'd30;
        4'd9: feature_size = 6'd32;
        default: feature_size = 6'd10;
    endcase
end

always @(*) begin
    init_lookup_addr = ((state == S_WRITE_START_REQ) || (state == S_WRITE_START_WRITE)) ? 10'd11 : addr_counter;
    ofmap_addr_offset = {{(MEM_AW-8){1'b0}}, ofmap_idx};
end

always @(*) begin
    final_ofmap_size_ext = {6'd0, feature_size} - 12'd4;
    ofmap_elem_count_ext = final_ofmap_size_ext * final_ofmap_size_ext;
    ofmap_word_count = (ofmap_elem_count_ext[9:0] + 10'd3) >> 2;
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        start_sync0 <= 1'b0;
        start_sync1 <= 1'b0;
        start_stable <= 1'b0;
        start_stable_d <= 1'b0;
        start_db_cnt <= 20'd0;
    end
    else begin
        start_sync0 <= start_bt;
        start_sync1 <= start_sync0;
        start_stable_d <= start_stable;
        if (start_sync1 == start_stable) begin
            start_db_cnt <= 20'd0;
        end
        else begin
            if (start_db_cnt == DEBOUNCE_MAX) begin
                start_stable <= start_sync1;
                start_db_cnt <= 20'd0;
            end
            else begin
                start_db_cnt <= start_db_cnt + 20'd1;
            end
        end
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        scan_counter <= 16'd0;
    end
    else begin
        scan_counter <= scan_counter + 16'd1;
    end
end

always @(*) begin
    case (tc)
        4'd0:  tc_bcd = 12'h000;
        4'd1:  tc_bcd = 12'h001;
        4'd2:  tc_bcd = 12'h002;
        4'd3:  tc_bcd = 12'h003;
        4'd4:  tc_bcd = 12'h004;
        4'd5:  tc_bcd = 12'h005;
        4'd6:  tc_bcd = 12'h006;
        4'd7:  tc_bcd = 12'h007;
        4'd8:  tc_bcd = 12'h008;
        4'd9:  tc_bcd = 12'h009;
        4'd10: tc_bcd = 12'h010;
        4'd11: tc_bcd = 12'h011;
        4'd12: tc_bcd = 12'h012;
        4'd13: tc_bcd = 12'h013;
        4'd14: tc_bcd = 12'h014;
        default: tc_bcd = 12'h000;
    endcase

    if (tc <= 4'd14) begin
        if (!result_valid[tc]) begin
            status_digit = 4'hF;
        end
        else begin
            status_digit = result_pass[tc] ? 4'h1 : 4'h0;
        end
    end
    else begin
        status_digit = 4'h0;
    end

    case (pass_count)
        4'd0:  total_pass_bcd = 16'h0000;
        4'd1:  total_pass_bcd = 16'h0001;
        4'd2:  total_pass_bcd = 16'h0002;
        4'd3:  total_pass_bcd = 16'h0003;
        4'd4:  total_pass_bcd = 16'h0004;
        4'd5:  total_pass_bcd = 16'h0005;
        4'd6:  total_pass_bcd = 16'h0006;
        4'd7:  total_pass_bcd = 16'h0007;
        4'd8:  total_pass_bcd = 16'h0008;
        4'd9:  total_pass_bcd = 16'h0009;
        4'd10: total_pass_bcd = 16'h0010;
        4'd11: total_pass_bcd = 16'h0011;
        4'd12: total_pass_bcd = 16'h0012;
        4'd13: total_pass_bcd = 16'h0013;
        4'd14: total_pass_bcd = 16'h0014;
        4'd15: total_pass_bcd = 16'h0015;
        default: total_pass_bcd = 16'h0000;
    endcase

    // Score mode format: 0000~0100, decimal score out of 100.
    // The score is always a multiple of 5, so use a small case table instead of / and %.
    case (weighted_score_decimal)
        8'd0:   score_display_bcd = 16'h0000;
        8'd5:   score_display_bcd = 16'h0005;
        8'd10:  score_display_bcd = 16'h0010;
        8'd15:  score_display_bcd = 16'h0015;
        8'd20:  score_display_bcd = 16'h0020;
        8'd25:  score_display_bcd = 16'h0025;
        8'd30:  score_display_bcd = 16'h0030;
        8'd35:  score_display_bcd = 16'h0035;
        8'd40:  score_display_bcd = 16'h0040;
        8'd45:  score_display_bcd = 16'h0045;
        8'd50:  score_display_bcd = 16'h0050;
        8'd55:  score_display_bcd = 16'h0055;
        8'd60:  score_display_bcd = 16'h0060;
        8'd65:  score_display_bcd = 16'h0065;
        8'd70:  score_display_bcd = 16'h0070;
        8'd75:  score_display_bcd = 16'h0075;
        8'd80:  score_display_bcd = 16'h0080;
        8'd85:  score_display_bcd = 16'h0085;
        8'd90:  score_display_bcd = 16'h0090;
        8'd95:  score_display_bcd = 16'h0095;
        8'd100: score_display_bcd = 16'h0100;
        default: score_display_bcd = 16'h0000;
    endcase

end

function [6:0] seven_seg_decode;
    input [3:0] nibble;
    begin
        case (nibble)
            4'h0: seven_seg_decode = 7'b1111110;
            4'h1: seven_seg_decode = 7'b0110000;
            4'h2: seven_seg_decode = 7'b1101101;
            4'h3: seven_seg_decode = 7'b1111001;
            4'h4: seven_seg_decode = 7'b0110011;
            4'h5: seven_seg_decode = 7'b1011011;
            4'h6: seven_seg_decode = 7'b1011111;
            4'h7: seven_seg_decode = 7'b1110000;
            4'h8: seven_seg_decode = 7'b1111111;
            4'h9: seven_seg_decode = 7'b1111011;
            4'ha: seven_seg_decode = 7'b1110111;
            4'hb: seven_seg_decode = 7'b0011111;
            4'hc: seven_seg_decode = 7'b1001110;
            4'hd: seven_seg_decode = 7'b0111101;
            4'he: seven_seg_decode = 7'b1001111;
            4'hf: seven_seg_decode = 7'b1000111;
            default: seven_seg_decode = 7'b0000000;
        endcase
    end
endfunction

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        display_value_q <= 16'h0000;
        scan_sel_q <= 2'd0;
        display_nibble_q <= 4'h0;
        anode_q <= 4'b0000;
        seven_seg_q <= 7'b0000000;
        seven_seg <= 7'b0000000;
        anode <= 4'b0000;
    end
    else begin
        // Pipeline stage for display source selection
        if (mode == 1'b1) begin
            display_value_q <= score_display_bcd;
        end
        else if (tc <= 4'd14) begin
            display_value_q <= {tc_bcd, status_digit};
        end
        else begin
            display_value_q <= total_pass_bcd;
        end
        scan_sel_q <= scan_sel;

        // Pipeline stage for scan-digit selection
        case (scan_sel_q)
            2'd0: begin
                display_nibble_q <= display_value_q[3:0];
                anode_q <= 4'b0001;
            end
            2'd1: begin
                display_nibble_q <= display_value_q[7:4];
                anode_q <= 4'b0010;
            end
            2'd2: begin
                display_nibble_q <= display_value_q[11:8];
                anode_q <= 4'b0100;
            end
            2'd3: begin
                display_nibble_q <= display_value_q[15:12];
                anode_q <= 4'b1000;
            end
            default: begin
                display_nibble_q <= 4'h0;
                anode_q <= 4'b0000;
            end
        endcase

        // Pipeline stage for seven-segment decode
        seven_seg_q <= seven_seg_decode(display_nibble_q);
        seven_seg <= seven_seg_q;
        anode <= anode_q;
    end
end

init_rom init_rom_inst (
    .clka(clk),
    .ena(1'b1),
    .addra(init_rom_addr),
    .douta(init_word)
);

golden_rom golden_rom_inst (
    .clka(clk),
    .ena(1'b1),
    .addra(golden_rom_addr),
    .douta(golden_word)
);

endmodule