`timescale 1ns / 1ps

module CNN(
    input         clk,
    input         rstn,
    input  [31:0] doutb,
    output reg        web,
    output reg        enb,
    output reg [31:0] dinb,
    output reg [9:0]  addr,
    output reg        done
);

localparam [9:0] INPUT_BASE  = 10'd16;
localparam [9:0] INTER_BASE  = 10'd512;
localparam [9:0] DEFAULT_OUT = 10'd272;
localparam [9:0] CNN_DONE_ADDR = 10'd10;
localparam [9:0] START_WAIT_CYCLES = 10'd64;
localparam ACC_W = 20;

localparam S_WAIT        = 5'd0;
localparam S_CFG_REQ     = 5'd1;
localparam S_CFG_CAP     = 5'd2;
localparam S_OUT_REQ     = 5'd3;
localparam S_OUT_CAP     = 5'd4;
localparam S_BIAS0_REQ   = 5'd5;
localparam S_BIAS0_CAP   = 5'd6;
localparam S_BIAS1_REQ   = 5'd7;
localparam S_BIAS1_CAP   = 5'd8;
localparam S_W_REQ       = 5'd9;
localparam S_W_CAP       = 5'd10;
localparam S_CONV1_INIT  = 5'd11;
localparam S_CONV2_INIT  = 5'd12;
localparam S_ROW_MUL     = 5'd13;
localparam S_LB_DRAIN    = 5'd14;
localparam S_CALC        = 5'd15;
localparam S_FIN_REQ     = 5'd17;
localparam S_DONE        = 5'd19;
localparam S_ADDR_CALC   = 5'd22;
localparam S_ROW_CAP     = 5'd23;
localparam S_ROW_XCAP    = 5'd24;
localparam S_ROW_FLUSH   = 5'd25;
localparam S_BUF_REQ     = 5'd26;
localparam S_BUF_CAP     = 5'd27;
localparam S_DOT         = 5'd28;
localparam S_SUM         = 5'd29;
localparam S_ROW_QUANT   = 5'd30;
localparam S_LB_QUANT    = 5'd31;

reg [4:0] state;
reg [9:0] wait_count;
reg layer;
reg [2:0] weight_addr;

reg [5:0] feature_size;
reg [5:0] in_size;
reg [5:0] out_size;
reg [9:0] total_elems;
reg [9:0] elem_idx;
reg [5:0] row;
reg [5:0] col;
reg [9:0] row_base;
reg [9:0] output_base;
reg [31:0] pack_word;
reg signed [ACC_W-1:0] acc;
reg signed [7:0] result_q;
reg signed [7:0] bias0;
reg signed [7:0] bias1;

reg signed [7:0] w0 [0:8];
reg signed [7:0] w1 [0:8];

reg signed [7:0] row_w0_c;
reg signed [7:0] row_w1_c;
reg signed [7:0] row_w2_c;
reg [9:0] read_addr_q;
reg [1:0] read_byte_sel_q;
reg [1:0] row_scan;
reg       cross_word_q;
reg [31:0] row_word0_q;
reg signed [ACC_W-1:0] prod0_q;
reg signed [ACC_W-1:0] prod1_q;
reg signed [ACC_W-1:0] prod2_q;
reg prod_valid_q;
reg signed [7:0] pix0_q;
reg signed [7:0] pix1_q;
reg signed [7:0] pix2_q;
reg signed [7:0] row_w0_q;
reg signed [7:0] row_w1_q;
reg signed [7:0] row_w2_q;

reg [31:0] row_buf0 [0:8];
reg [31:0] row_buf1 [0:8];
reg [31:0] row_buf2 [0:8];
reg [1:0]  row0_sel;
reg [1:0]  row1_sel;
reg [1:0]  row2_sel;
reg [1:0]  load_row_sel;
reg [3:0]  load_word_idx;
reg [3:0]  load_words_q;
reg [9:0] load_row_base;
reg signed [7:0] win00_q;
reg signed [7:0] win01_q;
reg signed [7:0] win02_q;
reg signed [7:0] win10_q;
reg signed [7:0] win11_q;
reg signed [7:0] win12_q;
reg signed [7:0] win20_q;
reg signed [7:0] win21_q;
reg signed [7:0] win22_q;
reg signed [ACC_W-1:0] dot0_q;
reg signed [ACC_W-1:0] dot1_q;
reg signed [ACC_W-1:0] dot2_q;
reg signed [ACC_W-1:0] dot3_q;
reg signed [ACC_W-1:0] dot4_q;
reg signed [ACC_W-1:0] dot5_q;
reg signed [ACC_W-1:0] dot6_q;
reg signed [ACC_W-1:0] dot7_q;
reg signed [ACC_W-1:0] dot8_q;
reg signed [ACC_W-1:0] sum0_q;
reg signed [ACC_W-1:0] sum1_q;
reg signed [ACC_W-1:0] sum2_q;
reg row_path_pixel;
reg pipe_launch_q;
reg dot_valid_q;
reg sum_valid_q;
reg [9:0] dot_elem_idx_q;
reg [9:0] sum_elem_idx_q;
reg dot_word_done_q;
reg sum_word_done_q;
reg dot_last_q;
reg sum_last_q;
reg dot_row_last_q;
reg sum_row_last_q;
reg quant_valid_q;
reg signed [7:0] quant_result_q;
reg [1:0] quant_elem_sel_q;
reg quant_word_done_q;
reg [9:0] quant_write_addr_q;
reg wb_valid_q;
reg [9:0] wb_addr_q;
reg [31:0] wb_data_q;
reg [4:0] drain_state_q;

wire signed [ACC_W-1:0] bias0_q12 = {{(ACC_W-8){bias0[7]}}, bias0} <<< 6;
wire signed [ACC_W-1:0] bias1_q12 = {{(ACC_W-8){bias1[7]}}, bias1} <<< 6;

wire last_elem = (elem_idx == (total_elems - 10'd1));
wire word_done = (elem_idx[1:0] == 2'd3) || last_elem;
wire [9:0] source_base = layer ? INTER_BASE : INPUT_BASE;
wire row_last_col = (col == (out_size - 6'd1));
wire [5:0] next_pixel_col = row_last_col ? 6'd0 : (col + 6'd1);
wire [9:0] next_pixel_row_base = row_last_col ? (row_base + {4'd0, in_size}) : row_base;
wire [9:0] next_pixel_first_index = next_pixel_row_base + {4'd0, next_pixel_col};
wire [9:0] next_read_addr = source_base + next_pixel_first_index[9:2];
wire [1:0] next_read_sel = next_pixel_first_index[1:0];
wire next_cross_word = next_pixel_first_index[1];
wire [9:0] curr_pixel_first_index = row_base + {4'd0, col};
wire [9:0] curr_read_addr = source_base + curr_pixel_first_index[9:2];
wire [1:0] curr_read_sel = curr_pixel_first_index[1:0];
wire curr_cross_word = curr_pixel_first_index[1];
wire [7:0] row_advance_sum = {6'd0, read_byte_sel_q} + {2'd0, in_size};
wire [9:0] row_advance_addr = read_addr_q + {5'd0, row_advance_sum[7:2]};
wire [1:0] row_advance_sel = row_advance_sum[1:0];
wire [31:0] dot_word0_c = (state == S_ROW_XCAP) ? row_word0_q : doutb;
wire [31:0] dot_word1_c = (state == S_ROW_XCAP) ? doutb : 32'd0;
wire signed [7:0] dot_pix0_c = get_window_byte(dot_word0_c, dot_word1_c, read_byte_sel_q, 2'd0);
wire signed [7:0] dot_pix1_c = get_window_byte(dot_word0_c, dot_word1_c, read_byte_sel_q, 2'd1);
wire signed [7:0] dot_pix2_c = get_window_byte(dot_word0_c, dot_word1_c, read_byte_sel_q, 2'd2);
wire signed [ACC_W-1:0] prod0_c = mul_q(dot_pix0_c, row_w0_c);
wire signed [ACC_W-1:0] prod1_c = mul_q(dot_pix1_c, row_w1_c);
wire signed [ACC_W-1:0] prod2_c = mul_q(dot_pix2_c, row_w2_c);
wire signed [ACC_W-1:0] prod_sum_q = prod0_q + prod1_q + prod2_q;
wire signed [ACC_W-1:0] final_acc_c = prod_valid_q ? (acc + prod_sum_q) : acc;
wire [9:0] next_top_row_base = row_base + {4'd0, in_size};
wire [9:0] next_bottom_load_base = row_base + {4'd0, in_size} + {3'd0, in_size, 1'b0};
wire [1:0] next_bottom_load_sel = next_bottom_load_base[1:0];
wire [3:0] next_bottom_load_words = row_words_for(next_bottom_load_sel, in_size);
wire [9:0] load_addr = source_base + load_row_base[9:2] + {6'd0, load_word_idx};
wire [3:0] load_word_idx_next = load_word_idx + 4'd1;
wire load_row_done = (load_word_idx == (load_words_q - 4'd1));
wire [9:0] load_next_row_base = load_row_base + {4'd0, in_size};
wire [1:0] load_next_row_sel = load_next_row_base[1:0];
wire signed [7:0] lb00_c = win00_q;
wire signed [7:0] lb01_c = win01_q;
wire signed [7:0] lb02_c = win02_q;
wire signed [7:0] lb10_c = win10_q;
wire signed [7:0] lb11_c = win11_q;
wire signed [7:0] lb12_c = win12_q;
wire signed [7:0] lb20_c = win20_q;
wire signed [7:0] lb21_c = win21_q;
wire signed [7:0] lb22_c = win22_q;
wire signed [7:0] dot_w0_c = layer ? w1[0] : w0[0];
wire signed [7:0] dot_w1_c = layer ? w1[1] : w0[1];
wire signed [7:0] dot_w2_c = layer ? w1[2] : w0[2];
wire signed [7:0] dot_w3_c = layer ? w1[3] : w0[3];
wire signed [7:0] dot_w4_c = layer ? w1[4] : w0[4];
wire signed [7:0] dot_w5_c = layer ? w1[5] : w0[5];
wire signed [7:0] dot_w6_c = layer ? w1[6] : w0[6];
wire signed [7:0] dot_w7_c = layer ? w1[7] : w0[7];
wire signed [7:0] dot_w8_c = layer ? w1[8] : w0[8];
wire signed [7:0] slide00_c = win01_q;
wire signed [7:0] slide01_c = win02_q;
wire signed [7:0] slide02_c = get_buf0_byte(col + 6'd3);
wire signed [7:0] slide10_c = win11_q;
wire signed [7:0] slide11_c = win12_q;
wire signed [7:0] slide12_c = get_buf1_byte(col + 6'd3);
wire signed [7:0] slide20_c = win21_q;
wire signed [7:0] slide21_c = win22_q;
wire signed [7:0] slide22_c = get_buf2_byte(col + 6'd3);
wire signed [ACC_W-1:0] conv_acc_c = sum0_q + sum1_q + sum2_q;
wire signed [7:0] conv_result_c = quantize_q12(conv_acc_c);
wire [31:0] pack_next = put_byte(pack_word, elem_idx[1:0], result_q);
wire [31:0] pipe_pack_next = put_byte(pack_word, sum_elem_idx_q[1:0], conv_result_c);
wire [31:0] quant_pack_next = put_byte(pack_word, quant_elem_sel_q, quant_result_q);
wire [9:0] pipe_write_addr = (layer ? output_base : INTER_BASE) + sum_elem_idx_q[9:2];

integer i;

function signed [ACC_W-1:0] mul_q;
    input signed [7:0] a;
    input signed [7:0] b;
    reg signed [15:0] product;
    begin
        product = a * b;
        mul_q = {{(ACC_W-16){product[15]}}, product};
    end
endfunction

function [9:0] row_offset_for;
    input [1:0] idx;
    begin
        case (idx)
            2'd0: row_offset_for = 10'd0;
            2'd1: row_offset_for = {4'd0, in_size};
            default: row_offset_for = {3'd0, in_size, 1'b0};
        endcase
    end
endfunction

function signed [7:0] get_byte;
    input [31:0] word;
    input [1:0] sel;
    begin
        case (sel)
            2'd0: get_byte = word[31:24];
            2'd1: get_byte = word[23:16];
            2'd2: get_byte = word[15:8];
            default: get_byte = word[7:0];
        endcase
    end
endfunction

function [3:0] row_words_for;
    input [1:0] start_sel;
    input [5:0] size;
    reg [7:0] span;
    begin
        span = {6'd0, start_sel} + {2'd0, size} + 8'd3;
        row_words_for = span[7:2];
    end
endfunction

function signed [7:0] get_buf0_byte;
    input [5:0] c;
    reg [7:0] pos;
    begin
        pos = {6'd0, row0_sel} + {2'd0, c};
        case (pos[5:2])
            4'd0: get_buf0_byte = get_byte(row_buf0[0], pos[1:0]);
            4'd1: get_buf0_byte = get_byte(row_buf0[1], pos[1:0]);
            4'd2: get_buf0_byte = get_byte(row_buf0[2], pos[1:0]);
            4'd3: get_buf0_byte = get_byte(row_buf0[3], pos[1:0]);
            4'd4: get_buf0_byte = get_byte(row_buf0[4], pos[1:0]);
            4'd5: get_buf0_byte = get_byte(row_buf0[5], pos[1:0]);
            4'd6: get_buf0_byte = get_byte(row_buf0[6], pos[1:0]);
            4'd7: get_buf0_byte = get_byte(row_buf0[7], pos[1:0]);
            default: get_buf0_byte = get_byte(row_buf0[8], pos[1:0]);
        endcase
    end
endfunction

function signed [7:0] get_buf1_byte;
    input [5:0] c;
    reg [7:0] pos;
    begin
        pos = {6'd0, row1_sel} + {2'd0, c};
        case (pos[5:2])
            4'd0: get_buf1_byte = get_byte(row_buf1[0], pos[1:0]);
            4'd1: get_buf1_byte = get_byte(row_buf1[1], pos[1:0]);
            4'd2: get_buf1_byte = get_byte(row_buf1[2], pos[1:0]);
            4'd3: get_buf1_byte = get_byte(row_buf1[3], pos[1:0]);
            4'd4: get_buf1_byte = get_byte(row_buf1[4], pos[1:0]);
            4'd5: get_buf1_byte = get_byte(row_buf1[5], pos[1:0]);
            4'd6: get_buf1_byte = get_byte(row_buf1[6], pos[1:0]);
            4'd7: get_buf1_byte = get_byte(row_buf1[7], pos[1:0]);
            default: get_buf1_byte = get_byte(row_buf1[8], pos[1:0]);
        endcase
    end
endfunction

function signed [7:0] get_buf2_byte;
    input [5:0] c;
    reg [7:0] pos;
    begin
        pos = {6'd0, row2_sel} + {2'd0, c};
        case (pos[5:2])
            4'd0: get_buf2_byte = get_byte(row_buf2[0], pos[1:0]);
            4'd1: get_buf2_byte = get_byte(row_buf2[1], pos[1:0]);
            4'd2: get_buf2_byte = get_byte(row_buf2[2], pos[1:0]);
            4'd3: get_buf2_byte = get_byte(row_buf2[3], pos[1:0]);
            4'd4: get_buf2_byte = get_byte(row_buf2[4], pos[1:0]);
            4'd5: get_buf2_byte = get_byte(row_buf2[5], pos[1:0]);
            4'd6: get_buf2_byte = get_byte(row_buf2[6], pos[1:0]);
            4'd7: get_buf2_byte = get_byte(row_buf2[7], pos[1:0]);
            default: get_buf2_byte = get_byte(row_buf2[8], pos[1:0]);
        endcase
    end
endfunction

function signed [7:0] get_window_byte;
    input [31:0] word0;
    input [31:0] word1;
    input [1:0] start_sel;
    input [1:0] tap_col;
    begin
        case (tap_col)
            2'd0: get_window_byte = get_byte(word0, start_sel);
            2'd1: get_window_byte = (start_sel == 2'd3) ?
                                     get_byte(word1, 2'd0) :
                                     get_byte(word0, start_sel + 2'd1);
            default: get_window_byte = (start_sel[1] == 1'b0) ?
                                       get_byte(word0, start_sel + 2'd2) :
                                       get_byte(word1, {1'b0, start_sel[0]});
        endcase
    end
endfunction

function [31:0] put_byte;
    input [31:0] word;
    input [1:0] sel;
    input [7:0] value;
    begin
        case (sel)
            2'd0: put_byte = {value, word[23:0]};
            2'd1: put_byte = {word[31:24], value, word[15:0]};
            2'd2: put_byte = {word[31:16], value, word[7:0]};
            default: put_byte = {word[31:8], value};
        endcase
    end
endfunction

function signed [7:0] quantize_q12;
    input signed [ACC_W-1:0] value;
    reg signed [ACC_W-1:0] shifted;
    reg [5:0] fraction;
    reg round_up;
    reg signed [ACC_W-1:0] rounded;
    begin
        shifted = value >>> 6;
        fraction = value[5:0];
        round_up = (fraction > 6'd32) ||
                   ((fraction == 6'd32) && (shifted[0] == 1'b1));
        rounded = shifted + {{(ACC_W-1){1'b0}}, round_up};

        if (rounded > 20'sd127) begin
            quantize_q12 = 8'sh7f;
        end
        else if (rounded < -20'sd128) begin
            quantize_q12 = -8'sd128;
        end
        else begin
            quantize_q12 = rounded[7:0];
        end
    end
endfunction

function [9:0] square_elems;
    input [5:0] side;
    begin
        case (side)
            6'd6:  square_elems = 10'd36;
            6'd8:  square_elems = 10'd64;
            6'd10: square_elems = 10'd100;
            6'd12: square_elems = 10'd144;
            6'd14: square_elems = 10'd196;
            6'd16: square_elems = 10'd256;
            6'd18: square_elems = 10'd324;
            6'd20: square_elems = 10'd400;
            6'd22: square_elems = 10'd484;
            6'd24: square_elems = 10'd576;
            6'd26: square_elems = 10'd676;
            6'd28: square_elems = 10'd784;
            6'd30: square_elems = 10'd900;
            default: square_elems = 10'd0;
        endcase
    end
endfunction

always @(*) begin
    case (row_scan)
        2'd0: begin
            row_w0_c = layer ? w1[0] : w0[0];
            row_w1_c = layer ? w1[1] : w0[1];
            row_w2_c = layer ? w1[2] : w0[2];
        end
        2'd1: begin
            row_w0_c = layer ? w1[3] : w0[3];
            row_w1_c = layer ? w1[4] : w0[4];
            row_w2_c = layer ? w1[5] : w0[5];
        end
        default: begin
            row_w0_c = layer ? w1[6] : w0[6];
            row_w1_c = layer ? w1[7] : w0[7];
            row_w2_c = layer ? w1[8] : w0[8];
        end
    endcase
end

always @(*) begin
    web = 1'b0;
    enb = 1'b0;
    dinb = 32'd0;
    addr = 10'd0;

    if (wb_valid_q) begin
        enb = 1'b1;
        web = 1'b1;
        addr = wb_addr_q;
        dinb = wb_data_q;
    end
    else begin
    case (state)
        S_CFG_REQ: begin
            enb = 1'b1;
            addr = 10'd12;
        end
        S_OUT_REQ: begin
            enb = 1'b1;
            addr = 10'd13;
        end
        S_BIAS0_REQ: begin
            enb = 1'b1;
            addr = 10'd14;
        end
        S_BIAS1_REQ: begin
            enb = 1'b1;
            addr = 10'd15;
        end
        S_W_REQ: begin
            enb = 1'b1;
            addr = {7'd0, weight_addr};
        end
        S_ADDR_CALC: begin
            enb = 1'b1;
            addr = read_addr_q;
        end
        S_ROW_CAP: begin
            if (cross_word_q) begin
                enb = 1'b1;
                addr = read_addr_q + 10'd1;
            end
        end
        S_ROW_MUL: begin
            if (row_scan != 2'd2) begin
                enb = 1'b1;
                addr = row_advance_addr;
            end
        end
        S_ROW_XCAP: begin
        end
        S_ROW_FLUSH: begin
            enb = 1'b0;
            addr = 10'd0;
        end
        S_BUF_REQ: begin
            enb = 1'b1;
            addr = load_addr;
        end
        S_BUF_CAP: begin
            if (!load_row_done) begin
                enb = 1'b1;
                addr = source_base + load_row_base[9:2] + {6'd0, load_word_idx_next};
            end
        end
        S_CALC: begin
            if (word_done) begin
                enb = 1'b1;
                web = 1'b1;
                addr = (layer ? output_base : INTER_BASE) + elem_idx[9:2];
                dinb = pack_next;
            end
            else if (layer) begin
                enb = 1'b1;
                addr = next_read_addr;
            end
        end
        S_LB_QUANT: begin
        end
        S_LB_DRAIN: begin
        end
        S_FIN_REQ: begin
            enb = 1'b1;
            web = 1'b1;
            addr = CNN_DONE_ADDR;
            dinb = 32'd1;
        end
        default: begin
            web = 1'b0;
            enb = 1'b0;
            dinb = 32'd0;
            addr = 10'd0;
        end
    endcase
    end
end

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= S_WAIT;
        wait_count <= 10'd0;
        done <= 1'b0;
        layer <= 1'b0;
        weight_addr <= 3'd0;
        feature_size <= 6'd0;
        in_size <= 6'd0;
        out_size <= 6'd0;
        total_elems <= 10'd0;
        elem_idx <= 10'd0;
        row <= 6'd0;
        col <= 6'd0;
        row_base <= 10'd0;
        output_base <= DEFAULT_OUT;
        pack_word <= 32'd0;
        acc <= {ACC_W{1'b0}};
        result_q <= 8'sd0;
        read_addr_q <= 10'd0;
        read_byte_sel_q <= 2'd0;
        row_scan <= 2'd0;
        cross_word_q <= 1'b0;
        row_word0_q <= 32'd0;
        prod0_q <= {ACC_W{1'b0}};
        prod1_q <= {ACC_W{1'b0}};
        prod2_q <= {ACC_W{1'b0}};
        prod_valid_q <= 1'b0;
        pix0_q <= 8'sd0;
        pix1_q <= 8'sd0;
        pix2_q <= 8'sd0;
        row_w0_q <= 8'sd0;
        row_w1_q <= 8'sd0;
        row_w2_q <= 8'sd0;
        row0_sel <= 2'd0;
        row1_sel <= 2'd0;
        row2_sel <= 2'd0;
        load_row_sel <= 2'd0;
        load_word_idx <= 4'd0;
        load_words_q <= 4'd0;
        load_row_base <= 10'd0;
        win00_q <= 8'sd0;
        win01_q <= 8'sd0;
        win02_q <= 8'sd0;
        win10_q <= 8'sd0;
        win11_q <= 8'sd0;
        win12_q <= 8'sd0;
        win20_q <= 8'sd0;
        win21_q <= 8'sd0;
        win22_q <= 8'sd0;
        dot0_q <= {ACC_W{1'b0}};
        dot1_q <= {ACC_W{1'b0}};
        dot2_q <= {ACC_W{1'b0}};
        dot3_q <= {ACC_W{1'b0}};
        dot4_q <= {ACC_W{1'b0}};
        dot5_q <= {ACC_W{1'b0}};
        dot6_q <= {ACC_W{1'b0}};
        dot7_q <= {ACC_W{1'b0}};
        dot8_q <= {ACC_W{1'b0}};
        sum0_q <= {ACC_W{1'b0}};
        sum1_q <= {ACC_W{1'b0}};
        sum2_q <= {ACC_W{1'b0}};
        row_path_pixel <= 1'b0;
        pipe_launch_q <= 1'b0;
        dot_valid_q <= 1'b0;
        sum_valid_q <= 1'b0;
        dot_elem_idx_q <= 10'd0;
        sum_elem_idx_q <= 10'd0;
        dot_word_done_q <= 1'b0;
        sum_word_done_q <= 1'b0;
        dot_last_q <= 1'b0;
        sum_last_q <= 1'b0;
        dot_row_last_q <= 1'b0;
        sum_row_last_q <= 1'b0;
        quant_valid_q <= 1'b0;
        quant_result_q <= 8'sd0;
        quant_elem_sel_q <= 2'd0;
        quant_word_done_q <= 1'b0;
        quant_write_addr_q <= 10'd0;
        wb_valid_q <= 1'b0;
        wb_addr_q <= 10'd0;
        wb_data_q <= 32'd0;
        drain_state_q <= S_WAIT;
        bias0 <= 8'sd0;
        bias1 <= 8'sd0;
        for (i = 0; i < 9; i = i + 1) begin
            w0[i] <= 8'sd0;
            w1[i] <= 8'sd0;
            row_buf0[i] <= 32'd0;
            row_buf1[i] <= 32'd0;
            row_buf2[i] <= 32'd0;
        end
    end
    else begin
        wb_valid_q <= 1'b0;
        case (state)
            S_WAIT: begin
                done <= 1'b0;
                if (wait_count == START_WAIT_CYCLES) begin
                    state <= S_CFG_REQ;
                end
                else begin
                    wait_count <= wait_count + 10'd1;
                end
            end

            S_CFG_REQ: begin
                state <= S_CFG_CAP;
            end

            S_CFG_CAP: begin
                feature_size <= doutb[6:1];
                state <= S_OUT_REQ;
            end

            S_OUT_REQ: begin
                state <= S_OUT_CAP;
            end

            S_OUT_CAP: begin
                output_base <= (doutb[10:1] == 10'd0) ? DEFAULT_OUT : doutb[10:1];
                state <= S_BIAS0_REQ;
            end

            S_BIAS0_REQ: begin
                state <= S_BIAS0_CAP;
            end

            S_BIAS0_CAP: begin
                bias0 <= doutb[7:0];
                state <= S_BIAS1_REQ;
            end

            S_BIAS1_REQ: begin
                state <= S_BIAS1_CAP;
            end

            S_BIAS1_CAP: begin
                bias1 <= doutb[7:0];
                weight_addr <= 3'd0;
                state <= S_W_REQ;
            end

            S_W_REQ: begin
                state <= S_W_CAP;
            end

            S_W_CAP: begin
                case (weight_addr)
                    3'd0: begin
                        w0[0] <= doutb[31:24];
                        w0[1] <= doutb[23:16];
                        w0[2] <= doutb[15:8];
                        w0[3] <= doutb[7:0];
                    end
                    3'd1: begin
                        w0[4] <= doutb[31:24];
                        w0[5] <= doutb[23:16];
                        w0[6] <= doutb[15:8];
                        w0[7] <= doutb[7:0];
                    end
                    3'd2: begin
                        w0[8] <= doutb[31:24];
                    end
                    3'd3: begin
                        w1[0] <= doutb[31:24];
                        w1[1] <= doutb[23:16];
                        w1[2] <= doutb[15:8];
                        w1[3] <= doutb[7:0];
                    end
                    3'd4: begin
                        w1[4] <= doutb[31:24];
                        w1[5] <= doutb[23:16];
                        w1[6] <= doutb[15:8];
                        w1[7] <= doutb[7:0];
                    end
                    default: begin
                        w1[8] <= doutb[31:24];
                    end
                endcase

                if (weight_addr == 3'd5) begin
                    state <= S_CONV1_INIT;
                end
                else begin
                    weight_addr <= weight_addr + 3'd1;
                    state <= S_W_REQ;
                end
            end

            S_CONV1_INIT: begin
                layer <= 1'b0;
                in_size <= feature_size;
                out_size <= feature_size - 6'd2;
                total_elems <= square_elems(feature_size - 6'd2);
                elem_idx <= 10'd0;
                row <= 6'd0;
                col <= 6'd0;
                row_base <= 10'd0;
                pack_word <= 32'd0;
                acc <= bias0_q12;
                prod_valid_q <= 1'b0;
                result_q <= 8'sd0;
                row_scan <= 2'd0;
                read_addr_q <= INPUT_BASE;
                read_byte_sel_q <= 2'd0;
                cross_word_q <= 1'b0;
                row0_sel <= 2'd0;
                row1_sel <= 2'd0;
                row2_sel <= 2'd0;
                load_row_sel <= 2'd0;
                load_word_idx <= 4'd0;
                load_row_base <= 10'd0;
                load_words_q <= row_words_for(2'd0, feature_size);
                row_path_pixel <= 1'b0;
                pipe_launch_q <= 1'b0;
                dot_valid_q <= 1'b0;
                sum_valid_q <= 1'b0;
                quant_valid_q <= 1'b0;
                wb_valid_q <= 1'b0;
                state <= S_BUF_REQ;
            end

            S_CONV2_INIT: begin
                layer <= 1'b1;
                in_size <= feature_size - 6'd2;
                out_size <= feature_size - 6'd4;
                total_elems <= square_elems(feature_size - 6'd4);
                elem_idx <= 10'd0;
                row <= 6'd0;
                col <= 6'd0;
                row_base <= 10'd0;
                pack_word <= 32'd0;
                acc <= bias1_q12;
                prod_valid_q <= 1'b0;
                result_q <= 8'sd0;
                row_scan <= 2'd0;
                read_addr_q <= INTER_BASE;
                read_byte_sel_q <= 2'd0;
                cross_word_q <= 1'b0;
                row0_sel <= 2'd0;
                row1_sel <= 2'd0;
                row2_sel <= 2'd0;
                load_row_sel <= 2'd0;
                load_word_idx <= 4'd0;
                load_row_base <= 10'd0;
                load_words_q <= row_words_for(2'd0, feature_size - 6'd2);
                row_path_pixel <= 1'b0;
                pipe_launch_q <= 1'b0;
                dot_valid_q <= 1'b0;
                sum_valid_q <= 1'b0;
                quant_valid_q <= 1'b0;
                wb_valid_q <= 1'b0;
                state <= S_BUF_REQ;
            end

            S_ADDR_CALC: begin
                prod_valid_q <= 1'b0;
                row_scan <= 2'd0;
                state <= S_ROW_CAP;
            end

            S_ROW_CAP: begin
                if (cross_word_q) begin
                    if (prod_valid_q) begin
                        acc <= acc + prod_sum_q;
                        prod_valid_q <= 1'b0;
                    end
                    row_word0_q <= doutb;
                    state <= S_ROW_XCAP;
                end
                else begin
                    if (prod_valid_q) begin
                        acc <= acc + prod_sum_q;
                        prod_valid_q <= 1'b0;
                    end
                    pix0_q <= dot_pix0_c;
                    pix1_q <= dot_pix1_c;
                    pix2_q <= dot_pix2_c;
                    row_w0_q <= row_w0_c;
                    row_w1_q <= row_w1_c;
                    row_w2_q <= row_w2_c;
                    state <= S_ROW_MUL;
                end
            end

            S_ROW_XCAP: begin
                if (prod_valid_q) begin
                    acc <= acc + prod_sum_q;
                    prod_valid_q <= 1'b0;
                end
                pix0_q <= dot_pix0_c;
                pix1_q <= dot_pix1_c;
                pix2_q <= dot_pix2_c;
                row_w0_q <= row_w0_c;
                row_w1_q <= row_w1_c;
                row_w2_q <= row_w2_c;
                state <= S_ROW_MUL;
            end

            S_ROW_MUL: begin
                prod0_q <= mul_q(pix0_q, row_w0_q);
                prod1_q <= mul_q(pix1_q, row_w1_q);
                prod2_q <= mul_q(pix2_q, row_w2_q);
                prod_valid_q <= 1'b1;
                if (row_scan == 2'd2) begin
                    state <= S_ROW_FLUSH;
                end
                else begin
                    row_scan <= row_scan + 2'd1;
                    read_addr_q <= row_advance_addr;
                    read_byte_sel_q <= row_advance_sel;
                    cross_word_q <= row_advance_sel[1];
                    state <= S_ROW_CAP;
                end
            end

            S_ROW_FLUSH: begin
                if (prod_valid_q) begin
                    acc <= acc + prod_sum_q;
                    prod_valid_q <= 1'b0;
                end
                state <= S_ROW_QUANT;
            end

            S_BUF_REQ: begin
                state <= S_BUF_CAP;
            end

            S_BUF_CAP: begin
                case (load_row_sel)
                    2'd0: row_buf0[load_word_idx] <= doutb;
                    2'd1: row_buf1[load_word_idx] <= doutb;
                    default: row_buf2[load_word_idx] <= doutb;
                endcase

                if (load_row_done) begin
                    load_word_idx <= 4'd0;
                    case (load_row_sel)
                        2'd0: begin
                            row0_sel <= load_row_base[1:0];
                            load_row_sel <= 2'd1;
                            load_row_base <= load_next_row_base;
                            load_words_q <= row_words_for(load_next_row_sel, in_size);
                            state <= S_BUF_REQ;
                        end
                        2'd1: begin
                            row1_sel <= load_row_base[1:0];
                            load_row_sel <= 2'd2;
                            load_row_base <= load_next_row_base;
                            load_words_q <= row_words_for(load_next_row_sel, in_size);
                            state <= S_BUF_REQ;
                        end
                        default: begin
                            row2_sel <= load_row_base[1:0];
                            row_path_pixel <= 1'b1;
                            pipe_launch_q <= 1'b0;
                            dot_valid_q <= 1'b0;
                            sum_valid_q <= 1'b0;
                            acc <= layer ? bias1_q12 : bias0_q12;
                            prod_valid_q <= 1'b0;
                            row_scan <= 2'd0;
                            read_addr_q <= curr_read_addr;
                            read_byte_sel_q <= curr_read_sel;
                            cross_word_q <= curr_cross_word;
                            state <= S_ADDR_CALC;
                        end
                    endcase
                end
                else begin
                    load_word_idx <= load_word_idx_next;
                    state <= S_BUF_CAP;
                end
            end

            S_DOT: begin
                dot0_q <= mul_q(lb00_c, dot_w0_c);
                dot1_q <= mul_q(lb01_c, dot_w1_c);
                dot2_q <= mul_q(lb02_c, dot_w2_c);
                dot3_q <= mul_q(lb10_c, dot_w3_c);
                dot4_q <= mul_q(lb11_c, dot_w4_c);
                dot5_q <= mul_q(lb12_c, dot_w5_c);
                dot6_q <= mul_q(lb20_c, dot_w6_c);
                dot7_q <= mul_q(lb21_c, dot_w7_c);
                dot8_q <= mul_q(lb22_c, dot_w8_c);
                dot_elem_idx_q <= elem_idx;
                dot_word_done_q <= word_done;
                dot_last_q <= last_elem;
                dot_row_last_q <= row_last_col;
                dot_valid_q <= 1'b1;

                if (last_elem || row_last_col) begin
                    pipe_launch_q <= 1'b0;
                end
                else begin
                    pipe_launch_q <= 1'b1;
                    elem_idx <= elem_idx + 10'd1;
                    win00_q <= slide00_c;
                    win01_q <= slide01_c;
                    win02_q <= slide02_c;
                    win10_q <= slide10_c;
                    win11_q <= slide11_c;
                    win12_q <= slide12_c;
                    win20_q <= slide20_c;
                    win21_q <= slide21_c;
                    win22_q <= slide22_c;
                    col <= col + 6'd1;
                end
                state <= S_SUM;
            end

            S_ROW_QUANT: begin
                result_q <= quantize_q12(acc);
                state <= S_CALC;
            end

            S_SUM: begin
                sum0_q <= dot0_q + dot1_q + dot2_q;
                sum1_q <= dot3_q + dot4_q + dot5_q;
                sum2_q <= dot6_q + dot7_q + dot8_q + (layer ? bias1_q12 : bias0_q12);
                sum_elem_idx_q <= dot_elem_idx_q;
                sum_word_done_q <= dot_word_done_q;
                sum_last_q <= dot_last_q;
                sum_row_last_q <= dot_row_last_q;
                sum_valid_q <= dot_valid_q;

                if (pipe_launch_q) begin
                    dot0_q <= mul_q(lb00_c, dot_w0_c);
                    dot1_q <= mul_q(lb01_c, dot_w1_c);
                    dot2_q <= mul_q(lb02_c, dot_w2_c);
                    dot3_q <= mul_q(lb10_c, dot_w3_c);
                    dot4_q <= mul_q(lb11_c, dot_w4_c);
                    dot5_q <= mul_q(lb12_c, dot_w5_c);
                    dot6_q <= mul_q(lb20_c, dot_w6_c);
                    dot7_q <= mul_q(lb21_c, dot_w7_c);
                    dot8_q <= mul_q(lb22_c, dot_w8_c);
                    dot_elem_idx_q <= elem_idx;
                    dot_word_done_q <= word_done;
                    dot_last_q <= last_elem;
                    dot_row_last_q <= row_last_col;
                    dot_valid_q <= 1'b1;

                    if (last_elem || row_last_col) begin
                        pipe_launch_q <= 1'b0;
                    end
                    else begin
                        elem_idx <= elem_idx + 10'd1;
                        win00_q <= slide00_c;
                        win01_q <= slide01_c;
                        win02_q <= slide02_c;
                        win10_q <= slide10_c;
                        win11_q <= slide11_c;
                        win12_q <= slide12_c;
                        win20_q <= slide20_c;
                        win21_q <= slide21_c;
                        win22_q <= slide22_c;
                        col <= col + 6'd1;
                    end
                end
                else begin
                    dot_valid_q <= 1'b0;
                end
                state <= S_LB_QUANT;
            end

            S_LB_QUANT: begin
                if (quant_valid_q) begin
                    if (quant_word_done_q) begin
                        wb_valid_q <= 1'b1;
                        wb_addr_q <= quant_write_addr_q;
                        wb_data_q <= quant_pack_next;
                        pack_word <= 32'd0;
                    end
                    else begin
                        pack_word <= quant_pack_next;
                    end
                end
                quant_valid_q <= 1'b0;

                if (sum_valid_q) begin
                    quant_valid_q <= 1'b1;
                    quant_result_q <= conv_result_c;
                    quant_elem_sel_q <= sum_elem_idx_q[1:0];
                    quant_word_done_q <= sum_word_done_q;
                    quant_write_addr_q <= pipe_write_addr;
                end

                if (sum_valid_q && sum_last_q) begin
                    dot_valid_q <= 1'b0;
                    sum_valid_q <= 1'b0;
                    pipe_launch_q <= 1'b0;
                    drain_state_q <= layer ? S_FIN_REQ : S_CONV2_INIT;
                    state <= S_LB_DRAIN;
                end
                else if (sum_valid_q && sum_row_last_q) begin
                    dot_valid_q <= 1'b0;
                    sum_valid_q <= 1'b0;
                    pipe_launch_q <= 1'b0;
                    elem_idx <= sum_elem_idx_q + 10'd1;
                    for (i = 0; i < 9; i = i + 1) begin
                        row_buf0[i] <= row_buf1[i];
                        row_buf1[i] <= row_buf2[i];
                    end
                    row0_sel <= row1_sel;
                    row1_sel <= row2_sel;
                    col <= 6'd0;
                    row <= row + 6'd1;
                    row_base <= next_top_row_base;
                    load_row_sel <= 2'd2;
                    load_word_idx <= 4'd0;
                    load_row_base <= next_bottom_load_base;
                    load_words_q <= next_bottom_load_words;
                    drain_state_q <= S_BUF_REQ;
                    state <= S_LB_DRAIN;
                end
                else begin
                    if (dot_valid_q) begin
                        sum0_q <= dot0_q + dot1_q + dot2_q;
                        sum1_q <= dot3_q + dot4_q + dot5_q;
                        sum2_q <= dot6_q + dot7_q + dot8_q + (layer ? bias1_q12 : bias0_q12);
                        sum_elem_idx_q <= dot_elem_idx_q;
                        sum_word_done_q <= dot_word_done_q;
                        sum_last_q <= dot_last_q;
                        sum_row_last_q <= dot_row_last_q;
                        sum_valid_q <= 1'b1;
                    end
                    else begin
                        sum_valid_q <= 1'b0;
                    end

                    if (pipe_launch_q) begin
                        dot0_q <= mul_q(lb00_c, dot_w0_c);
                        dot1_q <= mul_q(lb01_c, dot_w1_c);
                        dot2_q <= mul_q(lb02_c, dot_w2_c);
                        dot3_q <= mul_q(lb10_c, dot_w3_c);
                        dot4_q <= mul_q(lb11_c, dot_w4_c);
                        dot5_q <= mul_q(lb12_c, dot_w5_c);
                        dot6_q <= mul_q(lb20_c, dot_w6_c);
                        dot7_q <= mul_q(lb21_c, dot_w7_c);
                        dot8_q <= mul_q(lb22_c, dot_w8_c);
                        dot_elem_idx_q <= elem_idx;
                        dot_word_done_q <= word_done;
                        dot_last_q <= last_elem;
                        dot_row_last_q <= row_last_col;
                        dot_valid_q <= 1'b1;

                        if (last_elem || row_last_col) begin
                            pipe_launch_q <= 1'b0;
                        end
                        else begin
                            elem_idx <= elem_idx + 10'd1;
                            win00_q <= slide00_c;
                            win01_q <= slide01_c;
                            win02_q <= slide02_c;
                            win10_q <= slide10_c;
                            win11_q <= slide11_c;
                            win12_q <= slide12_c;
                            win20_q <= slide20_c;
                            win21_q <= slide21_c;
                            win22_q <= slide22_c;
                            col <= col + 6'd1;
                        end
                    end
                    else begin
                        dot_valid_q <= 1'b0;
                    end
                    state <= S_LB_QUANT;
                end
            end

            S_LB_DRAIN: begin
                if (quant_valid_q) begin
                    if (quant_word_done_q) begin
                        wb_valid_q <= 1'b1;
                        wb_addr_q <= quant_write_addr_q;
                        wb_data_q <= quant_pack_next;
                        pack_word <= 32'd0;
                        state <= S_LB_DRAIN;
                    end
                    else begin
                        pack_word <= quant_pack_next;
                        state <= drain_state_q;
                    end
                    quant_valid_q <= 1'b0;
                end
                else begin
                    state <= drain_state_q;
                end
            end

            S_CALC: begin
                if (1'b0) begin
                    if (word_done) begin
                        pack_word <= 32'd0;
                        if (last_elem) begin
                            state <= S_FIN_REQ;
                        end
                        else begin
                            read_addr_q <= next_read_addr;
                            read_byte_sel_q <= next_read_sel;
                            cross_word_q <= next_cross_word;
                            elem_idx <= elem_idx + 10'd1;
                            acc <= bias1_q12;
                            if (row_last_col) begin
                                col <= 6'd0;
                                row <= row + 6'd1;
                                row_base <= row_base + {4'd0, in_size};
                            end
                            else begin
                                col <= col + 6'd1;
                            end
                            state <= S_ADDR_CALC;
                        end
                    end
                    else begin
                        pack_word <= pack_next;
                        read_addr_q <= next_read_addr;
                        read_byte_sel_q <= next_read_sel;
                        cross_word_q <= next_cross_word;
                        row_scan <= 2'd0;
                        prod_valid_q <= 1'b0;
                        elem_idx <= elem_idx + 10'd1;
                        acc <= bias1_q12;
                        if (row_last_col) begin
                            col <= 6'd0;
                            row <= row + 6'd1;
                            row_base <= row_base + {4'd0, in_size};
                        end
                        else begin
                            col <= col + 6'd1;
                        end
                        state <= S_ROW_CAP;
                    end
                end
                else if (row_path_pixel) begin
                    row_path_pixel <= 1'b0;
                    if (word_done) begin
                        pack_word <= 32'd0;
                        if (last_elem) begin
                            state <= layer ? S_FIN_REQ : S_CONV2_INIT;
                        end
                        else begin
                            elem_idx <= elem_idx + 10'd1;
                            if (row_last_col) begin
                                for (i = 0; i < 9; i = i + 1) begin
                                    row_buf0[i] <= row_buf1[i];
                                    row_buf1[i] <= row_buf2[i];
                                end
                                row0_sel <= row1_sel;
                                row1_sel <= row2_sel;
                                col <= 6'd0;
                                row <= row + 6'd1;
                                row_base <= next_top_row_base;
                                load_row_sel <= 2'd2;
                                load_word_idx <= 4'd0;
                                load_row_base <= next_bottom_load_base;
                                load_words_q <= next_bottom_load_words;
                                state <= S_BUF_REQ;
                            end
                            else begin
                                win00_q <= get_buf0_byte(col + 6'd1);
                                win01_q <= get_buf0_byte(col + 6'd2);
                                win02_q <= get_buf0_byte(col + 6'd3);
                                win10_q <= get_buf1_byte(col + 6'd1);
                                win11_q <= get_buf1_byte(col + 6'd2);
                                win12_q <= get_buf1_byte(col + 6'd3);
                                win20_q <= get_buf2_byte(col + 6'd1);
                                win21_q <= get_buf2_byte(col + 6'd2);
                                win22_q <= get_buf2_byte(col + 6'd3);
                                col <= col + 6'd1;
                                state <= S_DOT;
                            end
                        end
                    end
                    else begin
                        pack_word <= pack_next;
                        elem_idx <= elem_idx + 10'd1;
                        if (row_last_col) begin
                            for (i = 0; i < 9; i = i + 1) begin
                                row_buf0[i] <= row_buf1[i];
                                row_buf1[i] <= row_buf2[i];
                            end
                            row0_sel <= row1_sel;
                            row1_sel <= row2_sel;
                            col <= 6'd0;
                            row <= row + 6'd1;
                            row_base <= next_top_row_base;
                            load_row_sel <= 2'd2;
                            load_word_idx <= 4'd0;
                            load_row_base <= next_bottom_load_base;
                            load_words_q <= next_bottom_load_words;
                            state <= S_BUF_REQ;
                        end
                        else begin
                            win00_q <= get_buf0_byte(col + 6'd1);
                            win01_q <= get_buf0_byte(col + 6'd2);
                            win02_q <= get_buf0_byte(col + 6'd3);
                            win10_q <= get_buf1_byte(col + 6'd1);
                            win11_q <= get_buf1_byte(col + 6'd2);
                            win12_q <= get_buf1_byte(col + 6'd3);
                            win20_q <= get_buf2_byte(col + 6'd1);
                            win21_q <= get_buf2_byte(col + 6'd2);
                            win22_q <= get_buf2_byte(col + 6'd3);
                            col <= col + 6'd1;
                            state <= S_DOT;
                        end
                    end
                end
                else if (word_done) begin
                    pack_word <= 32'd0;
                    if (last_elem) begin
                        state <= layer ? S_FIN_REQ : S_CONV2_INIT;
                    end
                    else begin
                        elem_idx <= elem_idx + 10'd1;
                        if (row_last_col) begin
                            for (i = 0; i < 9; i = i + 1) begin
                                row_buf0[i] <= row_buf1[i];
                                row_buf1[i] <= row_buf2[i];
                            end
                            row0_sel <= row1_sel;
                            row1_sel <= row2_sel;
                            col <= 6'd0;
                            row <= row + 6'd1;
                            row_base <= next_top_row_base;
                            load_row_sel <= 2'd2;
                            load_word_idx <= 4'd0;
                            load_row_base <= next_bottom_load_base;
                            load_words_q <= next_bottom_load_words;
                            state <= S_BUF_REQ;
                        end
                        else begin
                            win00_q <= win01_q;
                            win01_q <= win02_q;
                            win02_q <= get_buf0_byte(col + 6'd3);
                            win10_q <= win11_q;
                            win11_q <= win12_q;
                            win12_q <= get_buf1_byte(col + 6'd3);
                            win20_q <= win21_q;
                            win21_q <= win22_q;
                            win22_q <= get_buf2_byte(col + 6'd3);
                            col <= col + 6'd1;
                            state <= S_DOT;
                        end
                    end
                end
                else begin
                    pack_word <= pack_next;
                    elem_idx <= elem_idx + 10'd1;
                    if (row_last_col) begin
                        for (i = 0; i < 9; i = i + 1) begin
                            row_buf0[i] <= row_buf1[i];
                            row_buf1[i] <= row_buf2[i];
                        end
                        row0_sel <= row1_sel;
                        row1_sel <= row2_sel;
                        col <= 6'd0;
                        row <= row + 6'd1;
                        row_base <= next_top_row_base;
                        load_row_sel <= 2'd2;
                        load_word_idx <= 4'd0;
                        load_row_base <= next_bottom_load_base;
                        load_words_q <= next_bottom_load_words;
                        state <= S_BUF_REQ;
                    end
                    else begin
                        win00_q <= win01_q;
                        win01_q <= win02_q;
                        win02_q <= get_buf0_byte(col + 6'd3);
                        win10_q <= win11_q;
                        win11_q <= win12_q;
                        win12_q <= get_buf1_byte(col + 6'd3);
                        win20_q <= win21_q;
                        win21_q <= win22_q;
                        win22_q <= get_buf2_byte(col + 6'd3);
                        col <= col + 6'd1;
                        state <= S_DOT;
                    end
                end
            end

            S_FIN_REQ: begin
                done <= 1'b1;
                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1'b1;
            end

            default: begin
                state <= S_WAIT;
            end
        endcase
    end
end

endmodule
