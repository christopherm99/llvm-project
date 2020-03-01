set(LLVM_TARGET_DEFINITIONS X85.td)

# tablegen(LLVM X85GenAsmMatcher.inc -gen-asm-matcher)
# tablegen(LLVM X85GenAsmWriter.inc -gen-asm-writer)
# tablegen(LLVM X85GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
# tablegen(LLVM X85GenCallingConv.inc -gen-callingconv)
# tablegen(LLVM X85GenDAGISel.inc -gen-dag-isel)
# tablegen(LLVM X85GenDisassemblerTables.inc -gen-disassembler)
# tablegen(LLVM X85GenEVEX2VEXTables.inc -gen-x86-EVEX2VEX-tables)
# tablegen(LLVM X85GenExegesis.inc -gen-exegesis)
# tablegen(LLVM X85GenFastISel.inc -gen-fast-isel)
# tablegen(LLVM X85GenGlobalISel.inc -gen-global-isel)
tablegen(LLVM X85GenInstrInfo.inc -gen-instr-info)
# tablegen(LLVM X85GenRegisterBank.inc -gen-register-bank)
tablegen(LLVM X85GenRegisterInfo.inc -gen-register-info)
# tablegen(LLVM X85GenSubtargetInfo.inc -gen-subtarget)

# if (X85_GEN_FOLD_TABLES)
#   tablegen(LLVM X85GenFoldTables.inc -gen-x86-fold-tables)
# endif()

add_public_tablegen_target(X85CommonTableGen)

set(sources
  # X85AsmPrinter.cpp
  # X85AvoidTrailingCall.cpp
  # X85CallFrameOptimization.cpp
  # X85CallingConv.cpp
  # X85CallLowering.cpp
  # X85CmovConversion.cpp
  # X85CondBrFolding.cpp
  # X85DomainReassignment.cpp
  # X85DiscriminateMemOps.cpp
  # X85ExpandPseudo.cpp
  # X85FastISel.cpp
  # X85FixupBWInsts.cpp
  # X85FixupLEAs.cpp
  # X85AvoidStoreForwardingBlocks.cpp
  # X85FixupSetCC.cpp
  # X85FlagsCopyLowering.cpp
  # X85FloatingPoint.cpp
  # X85FrameLowering.cpp
  # X85InstructionSelector.cpp
  # X85ISelDAGToDAG.cpp
  # X85ISelLowering.cpp
  # X85IndirectBranchTracking.cpp
  # X85InterleavedAccess.cpp
  # X85InsertPrefetch.cpp
  # X85InstrFMA3Info.cpp
  # X85InstrFoldTables.cpp
  X85InstrInfo.cpp
  # X85EvexToVex.cpp
  # X85LegalizerInfo.cpp
  # X85MCInstLower.cpp
  # X85MachineFunctionInfo.cpp
  # X85MacroFusion.cpp
  # X85OptimizeLEAs.cpp
  # X85PadShortFunction.cpp
  # X85RegisterBankInfo.cpp
  X85RegisterInfo.cpp
  # X85RetpolineThunks.cpp
  # X85SelectionDAGInfo.cpp
  # X85ShuffleDecodeConstantPool.cpp
  # X85SpeculativeLoadHardening.cpp
  # X85Subtarget.cpp
  X85TargetMachine.cpp
  # X85TargetObjectFile.cpp
  # X85TargetTransformInfo.cpp
  # X85VZeroUpper.cpp
  # X85WinAllocaExpander.cpp
  # X85WinEHState.cpp
  )

add_llvm_target(X85CodeGen ${sources})

add_subdirectory(AsmParser)
# add_subdirectory(Disassembler)
add_subdirectory(TargetInfo)
# add_subdirectory(Utils)
