//===-- X85RegisterInfo.cpp - X85 Register Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the X85 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "X85RegisterInfo.h"
// #include "X85.h"
// #include "X85MachineFunctionInfo.h"
// #include "X85Subtarget.h"
#include "llvm/ADT/BitVector.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/CodeGen/MachineFrameInfo.h"
// #include "llvm/CodeGen/MachineFunction.h"
// #include "llvm/CodeGen/MachineInstrBuilder.h"
// #include "llvm/CodeGen/TargetInstrInfo.h"
// #include "llvm/IR/Type.h"
// #include "llvm/Support/CommandLine.h"
// #include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#define GET_REGINFO_ENUM
#include "X85GenRegisterInfo.inc"

// static cl::opt<bool>
// ReserveAppRegisters("sparc-reserve-app-registers", cl::Hidden, cl::init(false),
//                     cl::desc("Reserve application registers (%g2-%g4)"));

X85RegisterInfo::X85RegisterInfo() : X85GenRegisterInfo(X85::A) {}

// const MCPhysReg*
// X85RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
//   return CSR_SaveList;
// }

// const uint32_t *
// X85RegisterInfo::getCallPreservedMask(const MachineFunction &MF,
//                                         CallingConv::ID CC) const {
//   return CSR_RegMask;
// }

// const uint32_t*
// X85RegisterInfo::getRTCallPreservedMask(CallingConv::ID CC) const {
//   return RTCSR_RegMask;
// }

BitVector X85RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  Reserved.set(X85::SPH);
  Reserved.set(X85::SPL);
  Reserved.set(X85::SP);

  Reserved.set(X85::PCH);
  Reserved.set(X85::PCL);
  Reserved.set(X85::PC);

  return Reserved;
}

// const TargetRegisterClass*
// SparcRegisterInfo::getPointerRegClass(const MachineFunction &MF,
//                                       unsigned Kind) const {
//   const SparcSubtarget &Subtarget = MF.getSubtarget<SparcSubtarget>();
//   return Subtarget.is64Bit() ? &SP::I64RegsRegClass : &SP::IntRegsRegClass;
// }

// static void replaceFI(MachineFunction &MF, MachineBasicBlock::iterator II,
//                       MachineInstr &MI, const DebugLoc &dl,
//                       unsigned FIOperandNum, int Offset, unsigned FramePtr) {
//   // Replace frame index with a frame pointer reference.
//   if (Offset >= -4096 && Offset <= 4095) {
//     // If the offset is small enough to fit in the immediate field, directly
//     // encode it.
//     MI.getOperand(FIOperandNum).ChangeToRegister(FramePtr, false);
//     MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
//     return;
//   }

//   const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();

//   // FIXME: it would be better to scavenge a register here instead of
//   // reserving G1 all of the time.
//   if (Offset >= 0) {
//     // Emit nonnegaive immediates with sethi + or.
//     // sethi %hi(Offset), %g1
//     // add %g1, %fp, %g1
//     // Insert G1+%lo(offset) into the user.
//     BuildMI(*MI.getParent(), II, dl, TII.get(SP::SETHIi), SP::G1)
//       .addImm(HI22(Offset));


//     // Emit G1 = G1 + I6
//     BuildMI(*MI.getParent(), II, dl, TII.get(SP::ADDrr), SP::G1).addReg(SP::G1)
//       .addReg(FramePtr);
//     // Insert: G1+%lo(offset) into the user.
//     MI.getOperand(FIOperandNum).ChangeToRegister(SP::G1, false);
//     MI.getOperand(FIOperandNum + 1).ChangeToImmediate(LO10(Offset));
//     return;
//   }

//   // Emit Negative numbers with sethi + xor
//   // sethi %hix(Offset), %g1
//   // xor  %g1, %lox(offset), %g1
//   // add %g1, %fp, %g1
//   // Insert: G1 + 0 into the user.
//   BuildMI(*MI.getParent(), II, dl, TII.get(SP::SETHIi), SP::G1)
//     .addImm(HIX22(Offset));
//   BuildMI(*MI.getParent(), II, dl, TII.get(SP::XORri), SP::G1)
//     .addReg(SP::G1).addImm(LOX10(Offset));

//   BuildMI(*MI.getParent(), II, dl, TII.get(SP::ADDrr), SP::G1).addReg(SP::G1)
//     .addReg(FramePtr);
//   // Insert: G1+%lo(offset) into the user.
//   MI.getOperand(FIOperandNum).ChangeToRegister(SP::G1, false);
//   MI.getOperand(FIOperandNum + 1).ChangeToImmediate(0);
// }


// void
// SparcRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
//                                        int SPAdj, unsigned FIOperandNum,
//                                        RegScavenger *RS) const {
//   assert(SPAdj == 0 && "Unexpected");

//   MachineInstr &MI = *II;
//   DebugLoc dl = MI.getDebugLoc();
//   int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
//   MachineFunction &MF = *MI.getParent()->getParent();
//   const SparcSubtarget &Subtarget = MF.getSubtarget<SparcSubtarget>();
//   const SparcFrameLowering *TFI = getFrameLowering(MF);

//   unsigned FrameReg;
//   int Offset;
//   Offset = TFI->getFrameIndexReference(MF, FrameIndex, FrameReg);

//   Offset += MI.getOperand(FIOperandNum + 1).getImm();

//   if (!Subtarget.isV9() || !Subtarget.hasHardQuad()) {
//     if (MI.getOpcode() == SP::STQFri) {
//       const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
//       Register SrcReg = MI.getOperand(2).getReg();
//       Register SrcEvenReg = getSubReg(SrcReg, SP::sub_even64);
//       Register SrcOddReg = getSubReg(SrcReg, SP::sub_odd64);
//       MachineInstr *StMI =
//         BuildMI(*MI.getParent(), II, dl, TII.get(SP::STDFri))
//         .addReg(FrameReg).addImm(0).addReg(SrcEvenReg);
//       replaceFI(MF, *StMI, *StMI, dl, 0, Offset, FrameReg);
//       MI.setDesc(TII.get(SP::STDFri));
//       MI.getOperand(2).setReg(SrcOddReg);
//       Offset += 8;
//     } else if (MI.getOpcode() == SP::LDQFri) {
//       const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
//       Register DestReg = MI.getOperand(0).getReg();
//       Register DestEvenReg = getSubReg(DestReg, SP::sub_even64);
//       Register DestOddReg = getSubReg(DestReg, SP::sub_odd64);
//       MachineInstr *LdMI =
//         BuildMI(*MI.getParent(), II, dl, TII.get(SP::LDDFri), DestEvenReg)
//         .addReg(FrameReg).addImm(0);
//       replaceFI(MF, *LdMI, *LdMI, dl, 1, Offset, FrameReg);

//       MI.setDesc(TII.get(SP::LDDFri));
//       MI.getOperand(0).setReg(DestOddReg);
//       Offset += 8;
//     }
//   }

//   replaceFI(MF, II, MI, dl, FIOperandNum, Offset, FrameReg);

// }

// Register SparcRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
//   return SP::I6;
// }

// Sparc has no architectural need for stack realignment support,
// except that LLVM unfortunately currently implements overaligned
// stack objects by depending upon stack realignment support.
// If that ever changes, this can probably be deleted.
// bool SparcRegisterInfo::canRealignStack(const MachineFunction &MF) const {
//   if (!TargetRegisterInfo::canRealignStack(MF))
//     return false;

//   // Sparc always has a fixed frame pointer register, so don't need to
//   // worry about needing to reserve it. [even if we don't have a frame
//   // pointer for our frame, it still cannot be used for other things,
//   // or register window traps will be SADNESS.]

//   // If there's a reserved call frame, we can use SP to access locals.
//   if (getFrameLowering(MF)->hasReservedCallFrame(MF))
//     return true;

//   // Otherwise, we'd need a base pointer, but those aren't implemented
//   // for SPARC at the moment.

//   return false;
// }
