#include "llvm/CodeGen/TargetRegisterInfo.h"
#define GET_REGINFO_HEADER
#include "X85GenRegisterInfo.inc"

namespace llvm {
struct X85RegisterInfo : public X85GenRegisterInfo {
  X85RegisterInfo();

  /// Code Generation virtual methods...
  // const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  // const uint32_t *getCallPreservedMask(const MachineFunction &MF,
  //                                      CallingConv::ID CC) const override;

  // const uint32_t* getRTCallPreservedMask(CallingConv::ID CC) const;

  BitVector getReservedRegs(const MachineFunction &MF) const override;

  // const TargetRegisterClass *getPointerRegClass(const MachineFunction &MF,
  //                                               unsigned Kind) const override;

  // void eliminateFrameIndex(MachineBasicBlock::iterator II,
  //                          int SPAdj, unsigned FIOperandNum,
  //                          RegScavenger *RS = nullptr) const override;

  // Register getFrameRegister(const MachineFunction &MF) const override;

  // bool canRealignStack(const MachineFunction &MF) const override;
};

} // end namespace llvm
