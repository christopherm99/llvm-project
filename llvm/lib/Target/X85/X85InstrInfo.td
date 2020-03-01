class X85Inst<string asmstr, dag outs, dag ins, list<dag> pattern> : Instruction {
  let Namespace = "X85";

  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  let DecoderNamespace = "X85";
}

// def MEMrr : Operand<i16> {
//   let PrintMethod = "printMemOperand";
//   let MIOperandInfo = (ops IntRegs, IntRegs);
// }

// def MEMri : Operand<i16> {
//   let PrintMethod = "printMemOperand";
//   let MIOperandInfo = (ops ptr_rc, i16imm); // Not sure if this is supposed to i8imm?
// }

def MOVrr : X85Inst<"mov $dst, $src", (outs LowRegs:$dst), (ins LowRegs:$src), [(set i8:$dst, i8:$src)]>;                 // 01DDDSSS
def MOVrm : X85Inst<"mov $dst, [$addr]", (outs LowRegs:$dst), (ins MemOpnd:$addr), [(set i8:$dst, (OpNode addr:$addr))]>; // 01DDD110
def MOVmr : X85Inst<"mov [$addr], $dst", (outs MemOpnd:$addr), (ins LowRegs:$dst), [(set (OpNode addr:$addr), i8:$dst)]>; // 01110SSS
def MOVri : X85Inst<"mov $dst, $a", (outs LowRegs:$dst), (ins i8imm:$a), [(set i8:$dst, i8:$a)]>;                         // 00DDD110, data
def MOVmi : X85Inst<"mov [$addr], $a", (outs MemOpnd:$addr), (ins i8imm:$a), [(set (OpNode addr:$addr), i8:$a)]>;         // 00110110, data

def LXI : X85Inst<"lxi $dst, $a", (outs LargeRegs:$dst), (ins i16imm:$a), [(store i6:$dst, i16imm:$a)]>;    // 00RP0001, low data, high data

def LDA : X85Inst<"lda [$addr]", (), (ins i16imm:$addr), [(load)]>;

// class F_1<dag outs, dag ins, string asmstr, list<dag> pattern> : 
//   X85Inst<outs, ins, asmstr, pattern> {}