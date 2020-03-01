#include "X85TargetMachine.h"
#include "X85.h"
#include "TargetInfo/X85TargetInfo.h"
// #include "llvm/CodeGen/Passes.h"
// #include "llvm/CodeGen/TargetPassConfig.h"
// #include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

extern "C" void LLVMInitializeX85Target() {
  RegisterTarget<Triple::x85, /*HasJIT=*/false>
    X(getTheX85Target(), "x85", "X85");
}
