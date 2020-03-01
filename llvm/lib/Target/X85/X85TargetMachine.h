//===-- X85TargetMachine.h - Define TargetMachine for X85 ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the X85 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X85_X85TARGETMACHINE_H
#define LLVM_LIB_TARGET_X85_X85TARGETMACHINE_H

#include "X85InstrInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class Module;

class X85TargetMachine : public LLVMTargetMachine {
  const DataLayout DataLayout;       // Calculates type size & alignment
  X85Subtarget Subtarget;
  X85InstrInfo InstrInfo;
  TargetFrameInfo FrameInfo;

protected:
  virtual const TargetAsmInfo *createTargetAsmInfo() const;

public:
  X85TargetMachine(const Module &M, const std::string &FS);

  virtual const X85InstrInfo *getInstrInfo() const {return &InstrInfo; }
  virtual const TargetFrameInfo *getFrameInfo() const {return &FrameInfo; }
  virtual const TargetSubtarget *getSubtargetImpl() const{return &Subtarget; }
  virtual const TargetRegisterInfo *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  virtual const DataLayout *getDataLayout() const { return &DataLayout; }
  static unsigned getModuleMatchQuality(const Module &M);

  // Pass Pipeline Configuration
  // virtual bool addInstSelector(PassManagerBase &PM, bool Fast);
  // virtual bool addPreEmitPass(PassManagerBase &PM, bool Fast);
};

} // end namespace llvm

#endif
