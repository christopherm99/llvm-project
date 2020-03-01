//===-- X85AsmParser.cpp - Parse X85 assembly to MCInst instructions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X85MCExpr.h"
#include "MCTargetDesc/X85MCTargetDesc.h"
#include "TargetInfo/X85TargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>

using namespace llvm;

// The generated AsmMatcher X85GenAsmMatcher uses "X85" as the target
// namespace. But SPARC backend uses "SP" as its namespace.
namespace llvm {
namespace X85 {

    using namespace SP;

} // end namespace X85
} // end namespace llvm

namespace {

class X85Operand;

class X85AsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "X85GenAsmMatcher.inc"

  /// }

  // public interface of the MCTargetAsmParser.
  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;
  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;
  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;
  bool ParseDirective(AsmToken DirectiveID) override;

  unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                      unsigned Kind) override;

  // Custom parse functions for X85 specific operands.
  OperandMatchResultTy parseMEMOperand(OperandVector &Operands);

  OperandMatchResultTy parseMembarTag(OperandVector &Operands);

  OperandMatchResultTy parseOperand(OperandVector &Operands, StringRef Name);

  OperandMatchResultTy
  parseX85AsmOperand(std::unique_ptr<X85Operand> &Operand,
                       bool isCall = false);

  OperandMatchResultTy parseBranchModifiers(OperandVector &Operands);

  // Helper function for dealing with %lo / %hi in PIC mode.
  const X85MCExpr *adjustPICRelocation(X85MCExpr::VariantKind VK,
                                         const MCExpr *subExpr);

  // returns true if Tok is matched to a register and returns register in RegNo.
  bool matchRegisterName(const AsmToken &Tok, unsigned &RegNo,
                         unsigned &RegKind);

  bool matchX85AsmModifiers(const MCExpr *&EVal, SMLoc &EndLoc);

  bool is64Bit() const {
    return getSTI().getTargetTriple().getArch() == Triple::sparcv9;
  }

  bool expandSET(MCInst &Inst, SMLoc IDLoc,
                 SmallVectorImpl<MCInst> &Instructions);

public:
  X85AsmParser(const MCSubtargetInfo &sti, MCAsmParser &parser,
                const MCInstrInfo &MII,
                const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, sti, MII), Parser(parser) {
    Parser.addAliasForDirective(".half", ".2byte");
    Parser.addAliasForDirective(".uahalf", ".2byte");
    Parser.addAliasForDirective(".word", ".4byte");
    Parser.addAliasForDirective(".uaword", ".4byte");
    Parser.addAliasForDirective(".nword", is64Bit() ? ".8byte" : ".4byte");
    if (is64Bit())
      Parser.addAliasForDirective(".xword", ".8byte");

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
  }
};

} // end anonymous namespace

  static const MCPhysReg IntRegs[32] = {
    X85::G0, X85::G1, X85::G2, X85::G3,
    X85::G4, X85::G5, X85::G6, X85::G7,
    X85::O0, X85::O1, X85::O2, X85::O3,
    X85::O4, X85::O5, X85::O6, X85::O7,
    X85::L0, X85::L1, X85::L2, X85::L3,
    X85::L4, X85::L5, X85::L6, X85::L7,
    X85::I0, X85::I1, X85::I2, X85::I3,
    X85::I4, X85::I5, X85::I6, X85::I7 };

  static const MCPhysReg FloatRegs[32] = {
    X85::F0,  X85::F1,  X85::F2,  X85::F3,
    X85::F4,  X85::F5,  X85::F6,  X85::F7,
    X85::F8,  X85::F9,  X85::F10, X85::F11,
    X85::F12, X85::F13, X85::F14, X85::F15,
    X85::F16, X85::F17, X85::F18, X85::F19,
    X85::F20, X85::F21, X85::F22, X85::F23,
    X85::F24, X85::F25, X85::F26, X85::F27,
    X85::F28, X85::F29, X85::F30, X85::F31 };

  static const MCPhysReg DoubleRegs[32] = {
    X85::D0,  X85::D1,  X85::D2,  X85::D3,
    X85::D4,  X85::D5,  X85::D6,  X85::D7,
    X85::D8,  X85::D9,  X85::D10, X85::D11,
    X85::D12, X85::D13, X85::D14, X85::D15,
    X85::D16, X85::D17, X85::D18, X85::D19,
    X85::D20, X85::D21, X85::D22, X85::D23,
    X85::D24, X85::D25, X85::D26, X85::D27,
    X85::D28, X85::D29, X85::D30, X85::D31 };

  static const MCPhysReg QuadFPRegs[32] = {
    X85::Q0,  X85::Q1,  X85::Q2,  X85::Q3,
    X85::Q4,  X85::Q5,  X85::Q6,  X85::Q7,
    X85::Q8,  X85::Q9,  X85::Q10, X85::Q11,
    X85::Q12, X85::Q13, X85::Q14, X85::Q15 };

  static const MCPhysReg ASRRegs[32] = {
    SP::Y,     SP::ASR1,  SP::ASR2,  SP::ASR3,
    SP::ASR4,  SP::ASR5,  SP::ASR6, SP::ASR7,
    SP::ASR8,  SP::ASR9,  SP::ASR10, SP::ASR11,
    SP::ASR12, SP::ASR13, SP::ASR14, SP::ASR15,
    SP::ASR16, SP::ASR17, SP::ASR18, SP::ASR19,
    SP::ASR20, SP::ASR21, SP::ASR22, SP::ASR23,
    SP::ASR24, SP::ASR25, SP::ASR26, SP::ASR27,
    SP::ASR28, SP::ASR29, SP::ASR30, SP::ASR31};

  static const MCPhysReg IntPairRegs[] = {
    X85::G0_G1, X85::G2_G3, X85::G4_G5, X85::G6_G7,
    X85::O0_O1, X85::O2_O3, X85::O4_O5, X85::O6_O7,
    X85::L0_L1, X85::L2_L3, X85::L4_L5, X85::L6_L7,
    X85::I0_I1, X85::I2_I3, X85::I4_I5, X85::I6_I7};

  static const MCPhysReg CoprocRegs[32] = {
    X85::C0,  X85::C1,  X85::C2,  X85::C3,
    X85::C4,  X85::C5,  X85::C6,  X85::C7,
    X85::C8,  X85::C9,  X85::C10, X85::C11,
    X85::C12, X85::C13, X85::C14, X85::C15,
    X85::C16, X85::C17, X85::C18, X85::C19,
    X85::C20, X85::C21, X85::C22, X85::C23,
    X85::C24, X85::C25, X85::C26, X85::C27,
    X85::C28, X85::C29, X85::C30, X85::C31 };

  static const MCPhysReg CoprocPairRegs[] = {
    X85::C0_C1,   X85::C2_C3,   X85::C4_C5,   X85::C6_C7,
    X85::C8_C9,   X85::C10_C11, X85::C12_C13, X85::C14_C15,
    X85::C16_C17, X85::C18_C19, X85::C20_C21, X85::C22_C23,
    X85::C24_C25, X85::C26_C27, X85::C28_C29, X85::C30_C31};

namespace {

/// X85Operand - Instances of this class represent a parsed X85 machine
/// instruction.
class X85Operand : public MCParsedAsmOperand {
public:
  enum RegisterKind {
    rk_None,
    rk_IntReg,
    rk_IntPairReg,
    rk_FloatReg,
    rk_DoubleReg,
    rk_QuadReg,
    rk_CoprocReg,
    rk_CoprocPairReg,
    rk_Special,
  };

private:
  enum KindTy {
    k_Token,
    k_Register,
    k_Immediate,
    k_MemoryReg,
    k_MemoryImm
  } Kind;

  SMLoc StartLoc, EndLoc;

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
    RegisterKind Kind;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned Base;
    unsigned OffsetReg;
    const MCExpr *Off;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

public:
  X85Operand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  bool isToken() const override { return Kind == k_Token; }
  bool isReg() const override { return Kind == k_Register; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isMem() const override { return isMEMrr() || isMEMri(); }
  bool isMEMrr() const { return Kind == k_MemoryReg; }
  bool isMEMri() const { return Kind == k_MemoryImm; }
  bool isMembarTag() const { return Kind == k_Immediate; }

  bool isIntReg() const {
    return (Kind == k_Register && Reg.Kind == rk_IntReg);
  }

  bool isFloatReg() const {
    return (Kind == k_Register && Reg.Kind == rk_FloatReg);
  }

  bool isFloatOrDoubleReg() const {
    return (Kind == k_Register && (Reg.Kind == rk_FloatReg
                                   || Reg.Kind == rk_DoubleReg));
  }

  bool isCoprocReg() const {
    return (Kind == k_Register && Reg.Kind == rk_CoprocReg);
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  unsigned getReg() const override {
    assert((Kind == k_Register) && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert((Kind == k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  unsigned getMemBase() const {
    assert((Kind == k_MemoryReg || Kind == k_MemoryImm) && "Invalid access!");
    return Mem.Base;
  }

  unsigned getMemOffsetReg() const {
    assert((Kind == k_MemoryReg) && "Invalid access!");
    return Mem.OffsetReg;
  }

  const MCExpr *getMemOff() const {
    assert((Kind == k_MemoryImm) && "Invalid access!");
    return Mem.Off;
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const override {
    return StartLoc;
  }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const override {
    return EndLoc;
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case k_Token:     OS << "Token: " << getToken() << "\n"; break;
    case k_Register:  OS << "Reg: #" << getReg() << "\n"; break;
    case k_Immediate: OS << "Imm: " << getImm() << "\n"; break;
    case k_MemoryReg: OS << "Mem: " << getMemBase() << "+"
                         << getMemOffsetReg() << "\n"; break;
    case k_MemoryImm: assert(getMemOff() != nullptr);
      OS << "Mem: " << getMemBase()
         << "+" << *getMemOff()
         << "\n"; break;
    }
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const{
    // Add as immediate when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addMEMrrOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));

    assert(getMemOffsetReg() != 0 && "Invalid offset");
    Inst.addOperand(MCOperand::createReg(getMemOffsetReg()));
  }

  void addMEMriOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::createReg(getMemBase()));

    const MCExpr *Expr = getMemOff();
    addExpr(Inst, Expr);
  }

  void addMembarTagOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    addExpr(Inst, Expr);
  }

  static std::unique_ptr<X85Operand> CreateToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<X85Operand>(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<X85Operand> CreateReg(unsigned RegNum, unsigned Kind,
                                                 SMLoc S, SMLoc E) {
    auto Op = std::make_unique<X85Operand>(k_Register);
    Op->Reg.RegNum = RegNum;
    Op->Reg.Kind   = (X85Operand::RegisterKind)Kind;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<X85Operand> CreateImm(const MCExpr *Val, SMLoc S,
                                                 SMLoc E) {
    auto Op = std::make_unique<X85Operand>(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static bool MorphToIntPairReg(X85Operand &Op) {
    unsigned Reg = Op.getReg();
    assert(Op.Reg.Kind == rk_IntReg);
    unsigned regIdx = 32;
    if (Reg >= X85::G0 && Reg <= X85::G7)
      regIdx = Reg - X85::G0;
    else if (Reg >= X85::O0 && Reg <= X85::O7)
      regIdx = Reg - X85::O0 + 8;
    else if (Reg >= X85::L0 && Reg <= X85::L7)
      regIdx = Reg - X85::L0 + 16;
    else if (Reg >= X85::I0 && Reg <= X85::I7)
      regIdx = Reg - X85::I0 + 24;
    if (regIdx % 2 || regIdx > 31)
      return false;
    Op.Reg.RegNum = IntPairRegs[regIdx / 2];
    Op.Reg.Kind = rk_IntPairReg;
    return true;
  }

  static bool MorphToDoubleReg(X85Operand &Op) {
    unsigned Reg = Op.getReg();
    assert(Op.Reg.Kind == rk_FloatReg);
    unsigned regIdx = Reg - X85::F0;
    if (regIdx % 2 || regIdx > 31)
      return false;
    Op.Reg.RegNum = DoubleRegs[regIdx / 2];
    Op.Reg.Kind = rk_DoubleReg;
    return true;
  }

  static bool MorphToQuadReg(X85Operand &Op) {
    unsigned Reg = Op.getReg();
    unsigned regIdx = 0;
    switch (Op.Reg.Kind) {
    default: llvm_unreachable("Unexpected register kind!");
    case rk_FloatReg:
      regIdx = Reg - X85::F0;
      if (regIdx % 4 || regIdx > 31)
        return false;
      Reg = QuadFPRegs[regIdx / 4];
      break;
    case rk_DoubleReg:
      regIdx =  Reg - X85::D0;
      if (regIdx % 2 || regIdx > 31)
        return false;
      Reg = QuadFPRegs[regIdx / 2];
      break;
    }
    Op.Reg.RegNum = Reg;
    Op.Reg.Kind = rk_QuadReg;
    return true;
  }

  static bool MorphToCoprocPairReg(X85Operand &Op) {
    unsigned Reg = Op.getReg();
    assert(Op.Reg.Kind == rk_CoprocReg);
    unsigned regIdx = 32;
    if (Reg >= X85::C0 && Reg <= X85::C31)
      regIdx = Reg - X85::C0;
    if (regIdx % 2 || regIdx > 31)
      return false;
    Op.Reg.RegNum = CoprocPairRegs[regIdx / 2];
    Op.Reg.Kind = rk_CoprocPairReg;
    return true;
  }

  static std::unique_ptr<X85Operand>
  MorphToMEMrr(unsigned Base, std::unique_ptr<X85Operand> Op) {
    unsigned offsetReg = Op->getReg();
    Op->Kind = k_MemoryReg;
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = offsetReg;
    Op->Mem.Off = nullptr;
    return Op;
  }

  static std::unique_ptr<X85Operand>
  CreateMEMr(unsigned Base, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<X85Operand>(k_MemoryReg);
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = X85::G0;  // always 0
    Op->Mem.Off = nullptr;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<X85Operand>
  MorphToMEMri(unsigned Base, std::unique_ptr<X85Operand> Op) {
    const MCExpr *Imm  = Op->getImm();
    Op->Kind = k_MemoryImm;
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = 0;
    Op->Mem.Off = Imm;
    return Op;
  }
};

} // end anonymous namespace

bool X85AsmParser::expandSET(MCInst &Inst, SMLoc IDLoc,
                               SmallVectorImpl<MCInst> &Instructions) {
  MCOperand MCRegOp = Inst.getOperand(0);
  MCOperand MCValOp = Inst.getOperand(1);
  assert(MCRegOp.isReg());
  assert(MCValOp.isImm() || MCValOp.isExpr());

  // the imm operand can be either an expression or an immediate.
  bool IsImm = Inst.getOperand(1).isImm();
  int64_t RawImmValue = IsImm ? MCValOp.getImm() : 0;

  // Allow either a signed or unsigned 32-bit immediate.
  if (RawImmValue < -2147483648LL || RawImmValue > 4294967295LL) {
    return Error(IDLoc,
                 "set: argument must be between -2147483648 and 4294967295");
  }

  // If the value was expressed as a large unsigned number, that's ok.
  // We want to see if it "looks like" a small signed number.
  int32_t ImmValue = RawImmValue;
  // For 'set' you can't use 'or' with a negative operand on V9 because
  // that would splat the sign bit across the upper half of the destination
  // register, whereas 'set' is defined to zero the high 32 bits.
  bool IsEffectivelyImm13 =
      IsImm && ((is64Bit() ? 0 : -4096) <= ImmValue && ImmValue < 4096);
  const MCExpr *ValExpr;
  if (IsImm)
    ValExpr = MCConstantExpr::create(ImmValue, getContext());
  else
    ValExpr = MCValOp.getExpr();

  MCOperand PrevReg = MCOperand::createReg(X85::G0);

  // If not just a signed imm13 value, then either we use a 'sethi' with a
  // following 'or', or a 'sethi' by itself if there are no more 1 bits.
  // In either case, start with the 'sethi'.
  if (!IsEffectivelyImm13) {
    MCInst TmpInst;
    const MCExpr *Expr = adjustPICRelocation(X85MCExpr::VK_X85_HI, ValExpr);
    TmpInst.setLoc(IDLoc);
    TmpInst.setOpcode(SP::SETHIi);
    TmpInst.addOperand(MCRegOp);
    TmpInst.addOperand(MCOperand::createExpr(Expr));
    Instructions.push_back(TmpInst);
    PrevReg = MCRegOp;
  }

  // The low bits require touching in 3 cases:
  // * A non-immediate value will always require both instructions.
  // * An effectively imm13 value needs only an 'or' instruction.
  // * Otherwise, an immediate that is not effectively imm13 requires the
  //   'or' only if bits remain after clearing the 22 bits that 'sethi' set.
  // If the low bits are known zeros, there's nothing to do.
  // In the second case, and only in that case, must we NOT clear
  // bits of the immediate value via the %lo() assembler function.
  // Note also, the 'or' instruction doesn't mind a large value in the case
  // where the operand to 'set' was 0xFFFFFzzz - it does exactly what you mean.
  if (!IsImm || IsEffectivelyImm13 || (ImmValue & 0x3ff)) {
    MCInst TmpInst;
    const MCExpr *Expr;
    if (IsEffectivelyImm13)
      Expr = ValExpr;
    else
      Expr = adjustPICRelocation(X85MCExpr::VK_X85_LO, ValExpr);
    TmpInst.setLoc(IDLoc);
    TmpInst.setOpcode(SP::ORri);
    TmpInst.addOperand(MCRegOp);
    TmpInst.addOperand(PrevReg);
    TmpInst.addOperand(MCOperand::createExpr(Expr));
    Instructions.push_back(TmpInst);
  }
  return false;
}

bool X85AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                             OperandVector &Operands,
                                             MCStreamer &Out,
                                             uint64_t &ErrorInfo,
                                             bool MatchingInlineAsm) {
  MCInst Inst;
  SmallVector<MCInst, 8> Instructions;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                              MatchingInlineAsm);
  switch (MatchResult) {
  case Match_Success: {
    switch (Inst.getOpcode()) {
    default:
      Inst.setLoc(IDLoc);
      Instructions.push_back(Inst);
      break;
    case SP::SET:
      if (expandSET(Inst, IDLoc, Instructions))
        return true;
      break;
    }

    for (const MCInst &I : Instructions) {
      Out.EmitInstruction(I, getSTI());
    }
    return false;
  }

  case Match_MissingFeature:
    return Error(IDLoc,
                 "instruction requires a CPU feature not currently enabled");

  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((X85Operand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction mnemonic");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool X85AsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                   SMLoc &EndLoc) {
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  if (getLexer().getKind() != AsmToken::Percent)
    return false;
  Parser.Lex();
  unsigned regKind = X85Operand::rk_None;
  if (matchRegisterName(Tok, RegNo, regKind)) {
    Parser.Lex();
    return false;
  }

  return Error(StartLoc, "invalid register name");
}

static void applyMnemonicAliases(StringRef &Mnemonic,
                                 const FeatureBitset &Features,
                                 unsigned VariantID);

bool X85AsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                      StringRef Name, SMLoc NameLoc,
                                      OperandVector &Operands) {

  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(X85Operand::CreateToken(Name, NameLoc));

  // apply mnemonic aliases, if any, so that we can parse operands correctly.
  applyMnemonicAliases(Name, getAvailableFeatures(), 0);

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    // Read the first operand.
    if (getLexer().is(AsmToken::Comma)) {
      if (parseBranchModifiers(Operands) != MatchOperand_Success) {
        SMLoc Loc = getLexer().getLoc();
        return Error(Loc, "unexpected token");
      }
    }
    if (parseOperand(Operands, Name) != MatchOperand_Success) {
      SMLoc Loc = getLexer().getLoc();
      return Error(Loc, "unexpected token");
    }

    while (getLexer().is(AsmToken::Comma) || getLexer().is(AsmToken::Plus)) {
      if (getLexer().is(AsmToken::Plus)) {
      // Plus tokens are significant in software_traps (p83, sparcv8.pdf). We must capture them.
        Operands.push_back(X85Operand::CreateToken("+", Parser.getTok().getLoc()));
      }
      Parser.Lex(); // Eat the comma or plus.
      // Parse and remember the operand.
      if (parseOperand(Operands, Name) != MatchOperand_Success) {
        SMLoc Loc = getLexer().getLoc();
        return Error(Loc, "unexpected token");
      }
    }
  }
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    return Error(Loc, "unexpected token");
  }
  Parser.Lex(); // Consume the EndOfStatement.
  return false;
}

bool X85AsmParser::
ParseDirective(AsmToken DirectiveID)
{
  StringRef IDVal = DirectiveID.getString();

  if (IDVal == ".register") {
    // For now, ignore .register directive.
    Parser.eatToEndOfStatement();
    return false;
  }
  if (IDVal == ".proc") {
    // For compatibility, ignore this directive.
    // (It's supposed to be an "optimization" in the Sun assembler)
    Parser.eatToEndOfStatement();
    return false;
  }

  // Let the MC layer to handle other directives.
  return true;
}

OperandMatchResultTy
X85AsmParser::parseMEMOperand(OperandVector &Operands) {
  SMLoc S, E;
  unsigned BaseReg = 0;

  if (ParseRegister(BaseReg, S, E)) {
    return MatchOperand_NoMatch;
  }

  switch (getLexer().getKind()) {
  default: return MatchOperand_NoMatch;

  case AsmToken::Comma:
  case AsmToken::RBrac:
  case AsmToken::EndOfStatement:
    Operands.push_back(X85Operand::CreateMEMr(BaseReg, S, E));
    return MatchOperand_Success;

  case AsmToken:: Plus:
    Parser.Lex(); // Eat the '+'
    break;
  case AsmToken::Minus:
    break;
  }

  std::unique_ptr<X85Operand> Offset;
  OperandMatchResultTy ResTy = parseX85AsmOperand(Offset);
  if (ResTy != MatchOperand_Success || !Offset)
    return MatchOperand_NoMatch;

  Operands.push_back(
      Offset->isImm() ? X85Operand::MorphToMEMri(BaseReg, std::move(Offset))
                      : X85Operand::MorphToMEMrr(BaseReg, std::move(Offset)));

  return MatchOperand_Success;
}

OperandMatchResultTy X85AsmParser::parseMembarTag(OperandVector &Operands) {
  SMLoc S = Parser.getTok().getLoc();
  const MCExpr *EVal;
  int64_t ImmVal = 0;

  std::unique_ptr<X85Operand> Mask;
  if (parseX85AsmOperand(Mask) == MatchOperand_Success) {
    if (!Mask->isImm() || !Mask->getImm()->evaluateAsAbsolute(ImmVal) ||
        ImmVal < 0 || ImmVal > 127) {
      Error(S, "invalid membar mask number");
      return MatchOperand_ParseFail;
    }
  }

  while (getLexer().getKind() == AsmToken::Hash) {
    SMLoc TagStart = getLexer().getLoc();
    Parser.Lex(); // Eat the '#'.
    unsigned MaskVal = StringSwitch<unsigned>(Parser.getTok().getString())
      .Case("LoadLoad", 0x1)
      .Case("StoreLoad", 0x2)
      .Case("LoadStore", 0x4)
      .Case("StoreStore", 0x8)
      .Case("Lookaside", 0x10)
      .Case("MemIssue", 0x20)
      .Case("Sync", 0x40)
      .Default(0);

    Parser.Lex(); // Eat the identifier token.

    if (!MaskVal) {
      Error(TagStart, "unknown membar tag");
      return MatchOperand_ParseFail;
    }

    ImmVal |= MaskVal;

    if (getLexer().getKind() == AsmToken::Pipe)
      Parser.Lex(); // Eat the '|'.
  }

  EVal = MCConstantExpr::create(ImmVal, getContext());
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  Operands.push_back(X85Operand::CreateImm(EVal, S, E));
  return MatchOperand_Success;
}

OperandMatchResultTy
X85AsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {

  OperandMatchResultTy ResTy = MatchOperandParserImpl(Operands, Mnemonic);

  // If there wasn't a custom match, try the generic matcher below. Otherwise,
  // there was a match, but an error occurred, in which case, just return that
  // the operand parsing failed.
  if (ResTy == MatchOperand_Success || ResTy == MatchOperand_ParseFail)
    return ResTy;

  if (getLexer().is(AsmToken::LBrac)) {
    // Memory operand
    Operands.push_back(X85Operand::CreateToken("[",
                                                 Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the [

    if (Mnemonic == "cas" || Mnemonic == "casx" || Mnemonic == "casa") {
      SMLoc S = Parser.getTok().getLoc();
      if (getLexer().getKind() != AsmToken::Percent)
        return MatchOperand_NoMatch;
      Parser.Lex(); // eat %

      unsigned RegNo, RegKind;
      if (!matchRegisterName(Parser.getTok(), RegNo, RegKind))
        return MatchOperand_NoMatch;

      Parser.Lex(); // Eat the identifier token.
      SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer()-1);
      Operands.push_back(X85Operand::CreateReg(RegNo, RegKind, S, E));
      ResTy = MatchOperand_Success;
    } else {
      ResTy = parseMEMOperand(Operands);
    }

    if (ResTy != MatchOperand_Success)
      return ResTy;

    if (!getLexer().is(AsmToken::RBrac))
      return MatchOperand_ParseFail;

    Operands.push_back(X85Operand::CreateToken("]",
                                                 Parser.getTok().getLoc()));
    Parser.Lex(); // Eat the ]

    // Parse an optional address-space identifier after the address.
    if (getLexer().is(AsmToken::Integer)) {
      std::unique_ptr<X85Operand> Op;
      ResTy = parseX85AsmOperand(Op, false);
      if (ResTy != MatchOperand_Success || !Op)
        return MatchOperand_ParseFail;
      Operands.push_back(std::move(Op));
    }
    return MatchOperand_Success;
  }

  std::unique_ptr<X85Operand> Op;

  ResTy = parseX85AsmOperand(Op, (Mnemonic == "call"));
  if (ResTy != MatchOperand_Success || !Op)
    return MatchOperand_ParseFail;

  // Push the parsed operand into the list of operands
  Operands.push_back(std::move(Op));

  return MatchOperand_Success;
}

OperandMatchResultTy
X85AsmParser::parseX85AsmOperand(std::unique_ptr<X85Operand> &Op,
                                     bool isCall) {
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
  const MCExpr *EVal;

  Op = nullptr;
  switch (getLexer().getKind()) {
  default:  break;

  case AsmToken::Percent:
    Parser.Lex(); // Eat the '%'.
    unsigned RegNo;
    unsigned RegKind;
    if (matchRegisterName(Parser.getTok(), RegNo, RegKind)) {
      StringRef name = Parser.getTok().getString();
      Parser.Lex(); // Eat the identifier token.
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      switch (RegNo) {
      default:
        Op = X85Operand::CreateReg(RegNo, RegKind, S, E);
        break;
      case X85::PSR:
        Op = X85Operand::CreateToken("%psr", S);
        break;
      case X85::FSR:
        Op = X85Operand::CreateToken("%fsr", S);
        break;
      case X85::FQ:
        Op = X85Operand::CreateToken("%fq", S);
        break;
      case X85::CPSR:
        Op = X85Operand::CreateToken("%csr", S);
        break;
      case X85::CPQ:
        Op = X85Operand::CreateToken("%cq", S);
        break;
      case X85::WIM:
        Op = X85Operand::CreateToken("%wim", S);
        break;
      case X85::TBR:
        Op = X85Operand::CreateToken("%tbr", S);
        break;
      case X85::ICC:
        if (name == "xcc")
          Op = X85Operand::CreateToken("%xcc", S);
        else
          Op = X85Operand::CreateToken("%icc", S);
        break;
      }
      break;
    }
    if (matchX85AsmModifiers(EVal, E)) {
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      Op = X85Operand::CreateImm(EVal, S, E);
    }
    break;

  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::LParen:
  case AsmToken::Dot:
    if (!getParser().parseExpression(EVal, E))
      Op = X85Operand::CreateImm(EVal, S, E);
    break;

  case AsmToken::Identifier: {
    StringRef Identifier;
    if (!getParser().parseIdentifier(Identifier)) {
      E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);
      MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);

      const MCExpr *Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None,
                                                  getContext());
      X85MCExpr::VariantKind Kind = X85MCExpr::VK_X85_13;

      if (getContext().getObjectFileInfo()->isPositionIndependent()) {
        if (isCall)
          Kind = X85MCExpr::VK_X85_WPLT30;
        else
          Kind = X85MCExpr::VK_X85_GOT13;
      }

      Res = X85MCExpr::create(Kind, Res, getContext());

      Op = X85Operand::CreateImm(Res, S, E);
    }
    break;
  }
  }
  return (Op) ? MatchOperand_Success : MatchOperand_ParseFail;
}

OperandMatchResultTy
X85AsmParser::parseBranchModifiers(OperandVector &Operands) {
  // parse (,a|,pn|,pt)+

  while (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the comma

    if (!getLexer().is(AsmToken::Identifier))
      return MatchOperand_ParseFail;
    StringRef modName = Parser.getTok().getString();
    if (modName == "a" || modName == "pn" || modName == "pt") {
      Operands.push_back(X85Operand::CreateToken(modName,
                                                   Parser.getTok().getLoc()));
      Parser.Lex(); // eat the identifier.
    }
  }
  return MatchOperand_Success;
}

bool X85AsmParser::matchRegisterName(const AsmToken &Tok, unsigned &RegNo,
                                       unsigned &RegKind) {
  int64_t intVal = 0;
  RegNo = 0;
  RegKind = X85Operand::rk_None;
  if (Tok.is(AsmToken::Identifier)) {
    StringRef name = Tok.getString();

    // %fp
    if (name.equals("fp")) {
      RegNo = X85::I6;
      RegKind = X85Operand::rk_IntReg;
      return true;
    }
    // %sp
    if (name.equals("sp")) {
      RegNo = X85::O6;
      RegKind = X85Operand::rk_IntReg;
      return true;
    }

    if (name.equals("y")) {
      RegNo = X85::Y;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.substr(0, 3).equals_lower("asr")
        && !name.substr(3).getAsInteger(10, intVal)
        && intVal > 0 && intVal < 32) {
      RegNo = ASRRegs[intVal];
      RegKind = X85Operand::rk_Special;
      return true;
    }

    // %fprs is an alias of %asr6.
    if (name.equals("fprs")) {
      RegNo = ASRRegs[6];
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("icc")) {
      RegNo = X85::ICC;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("psr")) {
      RegNo = X85::PSR;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("fsr")) {
      RegNo = X85::FSR;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("fq")) {
      RegNo = X85::FQ;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("csr")) {
      RegNo = X85::CPSR;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("cq")) {
      RegNo = X85::CPQ;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("wim")) {
      RegNo = X85::WIM;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("tbr")) {
      RegNo = X85::TBR;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    if (name.equals("xcc")) {
      // FIXME:: check 64bit.
      RegNo = X85::ICC;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    // %fcc0 - %fcc3
    if (name.substr(0, 3).equals_lower("fcc")
        && !name.substr(3).getAsInteger(10, intVal)
        && intVal < 4) {
      // FIXME: check 64bit and  handle %fcc1 - %fcc3
      RegNo = X85::FCC0 + intVal;
      RegKind = X85Operand::rk_Special;
      return true;
    }

    // %g0 - %g7
    if (name.substr(0, 1).equals_lower("g")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[intVal];
      RegKind = X85Operand::rk_IntReg;
      return true;
    }
    // %o0 - %o7
    if (name.substr(0, 1).equals_lower("o")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[8 + intVal];
      RegKind = X85Operand::rk_IntReg;
      return true;
    }
    if (name.substr(0, 1).equals_lower("l")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[16 + intVal];
      RegKind = X85Operand::rk_IntReg;
      return true;
    }
    if (name.substr(0, 1).equals_lower("i")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 8) {
      RegNo = IntRegs[24 + intVal];
      RegKind = X85Operand::rk_IntReg;
      return true;
    }
    // %f0 - %f31
    if (name.substr(0, 1).equals_lower("f")
        && !name.substr(1, 2).getAsInteger(10, intVal) && intVal < 32) {
      RegNo = FloatRegs[intVal];
      RegKind = X85Operand::rk_FloatReg;
      return true;
    }
    // %f32 - %f62
    if (name.substr(0, 1).equals_lower("f")
        && !name.substr(1, 2).getAsInteger(10, intVal)
        && intVal >= 32 && intVal <= 62 && (intVal % 2 == 0)) {
      // FIXME: Check V9
      RegNo = DoubleRegs[intVal/2];
      RegKind = X85Operand::rk_DoubleReg;
      return true;
    }

    // %r0 - %r31
    if (name.substr(0, 1).equals_lower("r")
        && !name.substr(1, 2).getAsInteger(10, intVal) && intVal < 31) {
      RegNo = IntRegs[intVal];
      RegKind = X85Operand::rk_IntReg;
      return true;
    }

    // %c0 - %c31
    if (name.substr(0, 1).equals_lower("c")
        && !name.substr(1).getAsInteger(10, intVal)
        && intVal < 32) {
      RegNo = CoprocRegs[intVal];
      RegKind = X85Operand::rk_CoprocReg;
      return true;
    }

    if (name.equals("tpc")) {
      RegNo = X85::TPC;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tnpc")) {
      RegNo = X85::TNPC;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tstate")) {
      RegNo = X85::TSTATE;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tt")) {
      RegNo = X85::TT;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tick")) {
      RegNo = X85::TICK;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tba")) {
      RegNo = X85::TBA;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("pstate")) {
      RegNo = X85::PSTATE;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("tl")) {
      RegNo = X85::TL;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("pil")) {
      RegNo = X85::PIL;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("cwp")) {
      RegNo = X85::CWP;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("cansave")) {
      RegNo = X85::CANSAVE;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("canrestore")) {
      RegNo = X85::CANRESTORE;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("cleanwin")) {
      RegNo = X85::CLEANWIN;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("otherwin")) {
      RegNo = X85::OTHERWIN;
      RegKind = X85Operand::rk_Special;
      return true;
    }
    if (name.equals("wstate")) {
      RegNo = X85::WSTATE;
      RegKind = X85Operand::rk_Special;
      return true;
    }
  }
  return false;
}

// Determine if an expression contains a reference to the symbol
// "_GLOBAL_OFFSET_TABLE_".
static bool hasGOTReference(const MCExpr *Expr) {
  switch (Expr->getKind()) {
  case MCExpr::Target:
    if (const X85MCExpr *SE = dyn_cast<X85MCExpr>(Expr))
      return hasGOTReference(SE->getSubExpr());
    break;

  case MCExpr::Constant:
    break;

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(Expr);
    return hasGOTReference(BE->getLHS()) || hasGOTReference(BE->getRHS());
  }

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr &SymRef = *cast<MCSymbolRefExpr>(Expr);
    return (SymRef.getSymbol().getName() == "_GLOBAL_OFFSET_TABLE_");
  }

  case MCExpr::Unary:
    return hasGOTReference(cast<MCUnaryExpr>(Expr)->getSubExpr());
  }
  return false;
}

const X85MCExpr *
X85AsmParser::adjustPICRelocation(X85MCExpr::VariantKind VK,
                                    const MCExpr *subExpr) {
  // When in PIC mode, "%lo(...)" and "%hi(...)" behave differently.
  // If the expression refers contains _GLOBAL_OFFSETE_TABLE, it is
  // actually a %pc10 or %pc22 relocation. Otherwise, they are interpreted
  // as %got10 or %got22 relocation.

  if (getContext().getObjectFileInfo()->isPositionIndependent()) {
    switch(VK) {
    default: break;
    case X85MCExpr::VK_X85_LO:
      VK = (hasGOTReference(subExpr) ? X85MCExpr::VK_X85_PC10
                                     : X85MCExpr::VK_X85_GOT10);
      break;
    case X85MCExpr::VK_X85_HI:
      VK = (hasGOTReference(subExpr) ? X85MCExpr::VK_X85_PC22
                                     : X85MCExpr::VK_X85_GOT22);
      break;
    }
  }

  return X85MCExpr::create(VK, subExpr, getContext());
}

bool X85AsmParser::matchX85AsmModifiers(const MCExpr *&EVal,
                                            SMLoc &EndLoc) {
  AsmToken Tok = Parser.getTok();
  if (!Tok.is(AsmToken::Identifier))
    return false;

  StringRef name = Tok.getString();

  X85MCExpr::VariantKind VK = X85MCExpr::parseVariantKind(name);

  if (VK == X85MCExpr::VK_X85_None)
    return false;

  Parser.Lex(); // Eat the identifier.
  if (Parser.getTok().getKind() != AsmToken::LParen)
    return false;

  Parser.Lex(); // Eat the LParen token.
  const MCExpr *subExpr;
  if (Parser.parseParenExpression(subExpr, EndLoc))
    return false;

  EVal = adjustPICRelocation(VK, subExpr);
  return true;
}

extern "C" void LLVMInitializeX85AsmParser() {
  RegisterMCAsmParser<X85AsmParser> A(getTheX85Target());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "X85GenAsmMatcher.inc"

unsigned X85AsmParser::validateTargetOperandClass(MCParsedAsmOperand &GOp,
                                                    unsigned Kind) {
  X85Operand &Op = (X85Operand &)GOp;
  if (Op.isFloatOrDoubleReg()) {
    switch (Kind) {
    default: break;
    case MCK_DFPRegs:
      if (!Op.isFloatReg() || X85Operand::MorphToDoubleReg(Op))
        return MCTargetAsmParser::Match_Success;
      break;
    case MCK_QFPRegs:
      if (X85Operand::MorphToQuadReg(Op))
        return MCTargetAsmParser::Match_Success;
      break;
    }
  }
  if (Op.isIntReg() && Kind == MCK_IntPair) {
    if (X85Operand::MorphToIntPairReg(Op))
      return MCTargetAsmParser::Match_Success;
  }
  if (Op.isCoprocReg() && Kind == MCK_CoprocPair) {
     if (X85Operand::MorphToCoprocPairReg(Op))
       return MCTargetAsmParser::Match_Success;
   }
  return Match_InvalidOperand;
}
