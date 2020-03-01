//===-- X85TargetInfo.cpp - X85 Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/X85TargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheX85Target() {
  static Target TheX85Target;
  return TheX85Target;
}

extern "C" void LLVMInitializeX85TargetInfo() {
  RegisterTarget<Triple::X85, /*HasJIT=*/true> X(getTheX85Target(), "X85", "X85", "X85");
}
