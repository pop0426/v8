`timescale 1ns / 1ps

module Simple_CPU(
    input         CLK,
    input         RSTN,
    output reg        dmem_en,
    output reg        dmem_we,
    output reg [9:0]  dmem_addr,
    output reg [31:0] dmem_wdata,
    input      [31:0] dmem_rdata
);

localparam [6:0] OP_LOAD   = 7'b0000011;
localparam [6:0] OP_STORE  = 7'b0100011;
localparam [6:0] OP_IMM    = 7'b0010011;
localparam [6:0] OP_REG    = 7'b0110011;
localparam [6:0] OP_BRANCH = 7'b1100011;

reg [9:0]  pc_word;
reg [9:0]  fetch_pc_word_q;
reg        imem_valid_q;
reg        fetch_buf_valid;
reg [9:0]  fetch_buf_pc_word;
reg [31:0] fetch_buf_instr;

wire [31:0] instr;
wire [31:0] instr_mem_dout;

reg        if_id_valid;
reg [9:0]  if_id_pc_word;
reg [31:0] if_id_instr;

reg        id_ex_valid;
reg [9:0]  id_ex_pc_word;
reg [31:0] id_ex_rs1_val;
reg [31:0] id_ex_rs2_val;
reg [31:0] id_ex_imm;
reg [4:0]  id_ex_rs1;
reg [4:0]  id_ex_rs2;
reg [4:0]  id_ex_rd;
reg [2:0]  id_ex_funct3;
reg [6:0]  id_ex_funct7;
reg        id_ex_reg_write;
reg        id_ex_mem_read;
reg        id_ex_mem_write;
reg        id_ex_mem_to_reg;
reg        id_ex_alu_src_imm;
reg        id_ex_branch;
reg        id_ex_sub;

reg        ex_mem_valid;
reg        ex_mem_reg_write;
reg        ex_mem_mem_to_reg;
reg [4:0]  ex_mem_rd;
reg [31:0] ex_mem_alu_result;

reg        mem_wb_valid;
reg        mem_wb_reg_write;
reg        mem_wb_mem_to_reg;
reg [4:0]  mem_wb_rd;
reg [31:0] mem_wb_alu_result;
reg [31:0] mem_wb_load_data;

// Explicit flip-flop register file.
// x0 is hardwired to zero by read_reg(), and x1~x31 are ordinary registers.
reg [31:0] rf_x1,  rf_x2,  rf_x3,  rf_x4;
reg [31:0] rf_x5,  rf_x6,  rf_x7,  rf_x8;
reg [31:0] rf_x9,  rf_x10, rf_x11, rf_x12;
reg [31:0] rf_x13, rf_x14, rf_x15, rf_x16;
reg [31:0] rf_x17, rf_x18, rf_x19, rf_x20;
reg [31:0] rf_x21, rf_x22, rf_x23, rf_x24;
reg [31:0] rf_x25, rf_x26, rf_x27, rf_x28;
reg [31:0] rf_x29, rf_x30, rf_x31;

wire [6:0] id_opcode = if_id_instr[6:0];
wire [4:0] id_rd     = if_id_instr[11:7];
wire [2:0] id_funct3 = if_id_instr[14:12];
wire [4:0] id_rs1    = if_id_instr[19:15];
wire [4:0] id_rs2    = if_id_instr[24:20];
wire [6:0] id_funct7 = if_id_instr[31:25];

wire [31:0] id_imm_i = {{20{if_id_instr[31]}}, if_id_instr[31:20]};
wire [31:0] id_imm_s = {{20{if_id_instr[31]}}, if_id_instr[31:25], if_id_instr[11:7]};
wire [31:0] id_imm_b = {{19{if_id_instr[31]}}, if_id_instr[31], if_id_instr[7],
                         if_id_instr[30:25], if_id_instr[11:8], 1'b0};

wire id_is_load   = if_id_valid && (id_opcode == OP_LOAD)   && (id_funct3 == 3'b010);
wire id_is_store  = if_id_valid && (id_opcode == OP_STORE)  && (id_funct3 == 3'b010);
wire id_is_addi   = if_id_valid && (id_opcode == OP_IMM)    && (id_funct3 == 3'b000);
wire id_is_addsub = if_id_valid && (id_opcode == OP_REG)    && (id_funct3 == 3'b000) &&
                    ((id_funct7 == 7'b0000000) || (id_funct7 == 7'b0100000));
wire id_is_branch = if_id_valid && (id_opcode == OP_BRANCH) &&
                    ((id_funct3 == 3'b000) || (id_funct3 == 3'b100));

wire id_uses_rs1 = id_is_load || id_is_store || id_is_addi || id_is_addsub || id_is_branch;
wire id_uses_rs2 = id_is_store || id_is_addsub || id_is_branch;

wire [31:0] wb_wdata = mem_wb_mem_to_reg ? mem_wb_load_data : mem_wb_alu_result;
wire [31:0] id_rs1_raw = read_reg(id_rs1);
wire [31:0] id_rs2_raw = read_reg(id_rs2);
wire [31:0] id_rs1_value = (mem_wb_valid && mem_wb_reg_write &&
                            (mem_wb_rd != 5'd0) && (mem_wb_rd == id_rs1)) ?
                           wb_wdata : id_rs1_raw;
wire [31:0] id_rs2_value = (mem_wb_valid && mem_wb_reg_write &&
                            (mem_wb_rd != 5'd0) && (mem_wb_rd == id_rs2)) ?
                           wb_wdata : id_rs2_raw;

wire load_use_stall = id_ex_valid && id_ex_mem_read && (id_ex_rd != 5'd0) &&
                      ((id_uses_rs1 && (id_ex_rd == id_rs1)) ||
                       (id_uses_rs2 && (id_ex_rd == id_rs2)));

wire mem_ex_dep_stall = if_id_valid && (id_is_load || id_is_store) &&
                        id_ex_valid && id_ex_reg_write && (id_ex_rd != 5'd0) &&
                        (((id_is_load || id_is_store) && (id_ex_rd == id_rs1)) ||
                         (id_is_store && (id_ex_rd == id_rs2)));
wire decode_stall = load_use_stall || mem_ex_dep_stall;

reg [31:0] ex_rs1_fwd;
reg [31:0] ex_rs2_fwd;

always @(*) begin
    ex_rs1_fwd = id_ex_rs1_val;
    if (ex_mem_valid && ex_mem_reg_write && !ex_mem_mem_to_reg &&
        (ex_mem_rd != 5'd0) && (ex_mem_rd == id_ex_rs1)) begin
        ex_rs1_fwd = ex_mem_alu_result;
    end
    else if (mem_wb_valid && mem_wb_reg_write &&
             (mem_wb_rd != 5'd0) && (mem_wb_rd == id_ex_rs1)) begin
        ex_rs1_fwd = wb_wdata;
    end
end

always @(*) begin
    ex_rs2_fwd = id_ex_rs2_val;
    if (ex_mem_valid && ex_mem_reg_write && !ex_mem_mem_to_reg &&
        (ex_mem_rd != 5'd0) && (ex_mem_rd == id_ex_rs2)) begin
        ex_rs2_fwd = ex_mem_alu_result;
    end
    else if (mem_wb_valid && mem_wb_reg_write &&
             (mem_wb_rd != 5'd0) && (mem_wb_rd == id_ex_rs2)) begin
        ex_rs2_fwd = wb_wdata;
    end
end

wire [31:0] ex_alu_b = id_ex_alu_src_imm ? id_ex_imm : ex_rs2_fwd;
wire [31:0] ex_alu_result = id_ex_sub ? (ex_rs1_fwd - ex_alu_b) :
                                        (ex_rs1_fwd + ex_alu_b);
wire ex_beq_taken = (id_ex_funct3 == 3'b000) && (ex_rs1_fwd == ex_rs2_fwd);
wire ex_blt_taken = (id_ex_funct3 == 3'b100) && ($signed(ex_rs1_fwd) < $signed(ex_rs2_fwd));
wire branch_taken = id_ex_valid && id_ex_branch && (ex_beq_taken || ex_blt_taken);
wire signed [10:0] branch_base_word_s = {1'b0, id_ex_pc_word};
wire signed [10:0] branch_offset_word_s = id_ex_imm[12:2];
wire signed [10:0] branch_target_word_s = branch_base_word_s + branch_offset_word_s;
wire [9:0] branch_target_word = branch_target_word_s[9:0];

wire [31:0] mem_rs1_fwd = (mem_wb_valid && mem_wb_reg_write &&
                          (mem_wb_rd != 5'd0) && (mem_wb_rd == id_ex_rs1)) ?
                          wb_wdata : id_ex_rs1_val;
wire [31:0] mem_rs2_fwd = (mem_wb_valid && mem_wb_reg_write &&
                          (mem_wb_rd != 5'd0) && (mem_wb_rd == id_ex_rs2)) ?
                          wb_wdata : id_ex_rs2_val;
wire [31:0] mem_addr_result = mem_rs1_fwd + id_ex_imm;

function [31:0] read_reg;
    input [4:0] addr;
    begin
        case (addr)
            5'd1:  read_reg = rf_x1;
            5'd2:  read_reg = rf_x2;
            5'd3:  read_reg = rf_x3;
            5'd4:  read_reg = rf_x4;
            5'd5:  read_reg = rf_x5;
            5'd6:  read_reg = rf_x6;
            5'd7:  read_reg = rf_x7;
            5'd8:  read_reg = rf_x8;
            5'd9:  read_reg = rf_x9;
            5'd10: read_reg = rf_x10;
            5'd11: read_reg = rf_x11;
            5'd12: read_reg = rf_x12;
            5'd13: read_reg = rf_x13;
            5'd14: read_reg = rf_x14;
            5'd15: read_reg = rf_x15;
            5'd16: read_reg = rf_x16;
            5'd17: read_reg = rf_x17;
            5'd18: read_reg = rf_x18;
            5'd19: read_reg = rf_x19;
            5'd20: read_reg = rf_x20;
            5'd21: read_reg = rf_x21;
            5'd22: read_reg = rf_x22;
            5'd23: read_reg = rf_x23;
            5'd24: read_reg = rf_x24;
            5'd25: read_reg = rf_x25;
            5'd26: read_reg = rf_x26;
            5'd27: read_reg = rf_x27;
            5'd28: read_reg = rf_x28;
            5'd29: read_reg = rf_x29;
            5'd30: read_reg = rf_x30;
            5'd31: read_reg = rf_x31;
            default: read_reg = 32'd0;
        endcase
