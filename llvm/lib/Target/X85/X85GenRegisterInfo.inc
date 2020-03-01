/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Target Register Enum Values                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_REGINFO_ENUM
#undef GET_REGINFO_ENUM

namespace llvm {

class MCRegisterClass;
extern const MCRegisterClass X85MCRegisterClasses[];

namespace X85 {
enum {
  NoRegister,
  A = 1,
  B = 2,
  C = 3,
  D = 4,
  E = 5,
  H = 6,
  HL = 7,
  L = 8,
  PC = 9,
  PCH = 10,
  PCL = 11,
  SP = 12,
  SPH = 13,
  SPL = 14,
  NUM_TARGET_REGS 	// 15
};
} // end namespace X85

// Register classes

namespace X85 {
enum {
  LowRegsRegClassID = 0,
  HighRegsRegClassID = 1,
  LargeRegsRegClassID = 2,

  };
} // end namespace X85


// Subregister indices

enum {
  NoSubRegister,
  anonymous_2225,	// 1
  anonymous_2226,	// 2
  NUM_TARGET_SUBREGS
};
} // end namespace llvm

#endif // GET_REGINFO_ENUM

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* MC Register Information                                                    *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_REGINFO_MC_DESC
#undef GET_REGINFO_MC_DESC

namespace llvm {

extern const MCPhysReg X85RegDiffLists[] = {
  /* 0 */ 1, 1, 0,
  /* 3 */ 5, 1, 0,
  /* 6 */ 7, 1, 0,
  /* 9 */ 9, 1, 0,
  /* 12 */ 65535, 2, 0,
  /* 15 */ 6, 0,
  /* 17 */ 65532, 0,
  /* 19 */ 65533, 0,
  /* 21 */ 65534, 0,
  /* 23 */ 65535, 0,
};

extern const LaneBitmask X85LaneMaskLists[] = {
  /* 0 */ LaneBitmask(0x00000000), LaneBitmask::getAll(),
  /* 2 */ LaneBitmask(0x00000001), LaneBitmask(0x00000002), LaneBitmask::getAll(),
};

extern const uint16_t X85SubRegIdxLists[] = {
  /* 0 */ 1, 2, 0,
};

extern const MCRegisterInfo::SubRegCoveredBits X85SubRegIdxRanges[] = {
  { 65535, 65535 },
  { 0, 8 },	// anonymous_2225
  { 8, 8 },	// anonymous_2226
};

extern const char X85RegStrings[] = {
  /* 0 */ 'A', 0,
  /* 2 */ 'B', 0,
  /* 4 */ 'P', 'C', 0,
  /* 7 */ 'D', 0,
  /* 9 */ 'E', 0,
  /* 11 */ 'P', 'C', 'H', 0,
  /* 15 */ 'S', 'P', 'H', 0,
  /* 19 */ 'P', 'C', 'L', 0,
  /* 23 */ 'H', 'L', 0,
  /* 26 */ 'S', 'P', 'L', 0,
  /* 30 */ 'S', 'P', 0,
};

extern const MCRegisterDesc X85RegDesc[] = { // Descriptors
  { 1, 0, 0, 0, 0, 0 },
  { 0, 2, 2, 2, 369, 0 },
  { 2, 2, 2, 2, 369, 0 },
  { 5, 2, 2, 2, 369, 0 },
  { 7, 2, 2, 2, 369, 0 },
  { 9, 2, 2, 2, 369, 0 },
  { 13, 2, 1, 2, 369, 0 },
  { 23, 12, 2, 0, 48, 2 },
  { 21, 2, 23, 2, 240, 0 },
  { 4, 0, 2, 0, 96, 2 },
  { 11, 2, 23, 2, 305, 0 },
  { 19, 2, 21, 2, 305, 0 },
  { 30, 0, 2, 0, 144, 2 },
  { 15, 2, 23, 2, 273, 0 },
  { 26, 2, 21, 2, 273, 0 },
};

extern const MCPhysReg X85RegUnitRoots[][2] = {
  { X85::A },
  { X85::B },
  { X85::C },
  { X85::D },
  { X85::E },
  { X85::H },
  { X85::L },
  { X85::PCH },
  { X85::PCL },
  { X85::SPH },
  { X85::SPL },
};

namespace {     // Register classes...
  // LowRegs Register Class...
  const MCPhysReg LowRegs[] = {
    X85::A, X85::B, X85::C, X85::D, X85::E, X85::L, X85::SPL, X85::PCL, 
  };

  // LowRegs Bit set.
  const uint8_t LowRegsBits[] = {
    0x3e, 0x49, 
  };

  // HighRegs Register Class...
  const MCPhysReg HighRegs[] = {
    X85::H, X85::SPH, X85::PCH, 
  };

  // HighRegs Bit set.
  const uint8_t HighRegsBits[] = {
    0x40, 0x24, 
  };

  // LargeRegs Register Class...
  const MCPhysReg LargeRegs[] = {
    X85::HL, X85::SP, X85::PC, 
  };

  // LargeRegs Bit set.
  const uint8_t LargeRegsBits[] = {
    0x80, 0x12, 
  };

} // end anonymous namespace

extern const char X85RegClassStrings[] = {
  /* 0 */ 'L', 'a', 'r', 'g', 'e', 'R', 'e', 'g', 's', 0,
  /* 10 */ 'H', 'i', 'g', 'h', 'R', 'e', 'g', 's', 0,
  /* 19 */ 'L', 'o', 'w', 'R', 'e', 'g', 's', 0,
};

extern const MCRegisterClass X85MCRegisterClasses[] = {
  { LowRegs, LowRegsBits, 19, 8, sizeof(LowRegsBits), X85::LowRegsRegClassID, 1, true },
  { HighRegs, HighRegsBits, 10, 3, sizeof(HighRegsBits), X85::HighRegsRegClassID, 1, true },
  { LargeRegs, LargeRegsBits, 0, 3, sizeof(LargeRegsBits), X85::LargeRegsRegClassID, 1, true },
};

// X85 Dwarf<->LLVM register mappings.
extern const MCRegisterInfo::DwarfLLVMRegPair X85DwarfFlavour0Dwarf2L[] = {
  { 0U, X85::A },
  { 1U, X85::B },
  { 2U, X85::C },
  { 3U, X85::D },
  { 4U, X85::E },
  { 5U, X85::H },
  { 6U, X85::L },
  { 7U, X85::SPL },
  { 8U, X85::SPH },
  { 9U, X85::PCL },
  { 10U, X85::PCH },
  { 11U, X85::HL },
  { 12U, X85::SP },
  { 13U, X85::PC },
};
extern const unsigned X85DwarfFlavour0Dwarf2LSize = array_lengthof(X85DwarfFlavour0Dwarf2L);

extern const MCRegisterInfo::DwarfLLVMRegPair X85EHFlavour0Dwarf2L[] = {
  { 0U, X85::A },
  { 1U, X85::B },
  { 2U, X85::C },
  { 3U, X85::D },
  { 4U, X85::E },
  { 5U, X85::H },
  { 6U, X85::L },
  { 7U, X85::SPL },
  { 8U, X85::SPH },
  { 9U, X85::PCL },
  { 10U, X85::PCH },
  { 11U, X85::HL },
  { 12U, X85::SP },
  { 13U, X85::PC },
};
extern const unsigned X85EHFlavour0Dwarf2LSize = array_lengthof(X85EHFlavour0Dwarf2L);

extern const MCRegisterInfo::DwarfLLVMRegPair X85DwarfFlavour0L2Dwarf[] = {
  { X85::A, 0U },
  { X85::B, 1U },
  { X85::C, 2U },
  { X85::D, 3U },
  { X85::E, 4U },
  { X85::H, 5U },
  { X85::HL, 11U },
  { X85::L, 6U },
  { X85::PC, 13U },
  { X85::PCH, 10U },
  { X85::PCL, 9U },
  { X85::SP, 12U },
  { X85::SPH, 8U },
  { X85::SPL, 7U },
};
extern const unsigned X85DwarfFlavour0L2DwarfSize = array_lengthof(X85DwarfFlavour0L2Dwarf);

extern const MCRegisterInfo::DwarfLLVMRegPair X85EHFlavour0L2Dwarf[] = {
  { X85::A, 0U },
  { X85::B, 1U },
  { X85::C, 2U },
  { X85::D, 3U },
  { X85::E, 4U },
  { X85::H, 5U },
  { X85::HL, 11U },
  { X85::L, 6U },
  { X85::PC, 13U },
  { X85::PCH, 10U },
  { X85::PCL, 9U },
  { X85::SP, 12U },
  { X85::SPH, 8U },
  { X85::SPL, 7U },
};
extern const unsigned X85EHFlavour0L2DwarfSize = array_lengthof(X85EHFlavour0L2Dwarf);

extern const uint16_t X85RegEncodingTable[] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};
static inline void InitX85MCRegisterInfo(MCRegisterInfo *RI, unsigned RA, unsigned DwarfFlavour = 0, unsigned EHFlavour = 0, unsigned PC = 0) {
  RI->InitMCRegisterInfo(X85RegDesc, 15, RA, PC, X85MCRegisterClasses, 3, X85RegUnitRoots, 11, X85RegDiffLists, X85LaneMaskLists, X85RegStrings, X85RegClassStrings, X85SubRegIdxLists, 3,
X85SubRegIdxRanges, X85RegEncodingTable);

  switch (DwarfFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    RI->mapDwarfRegsToLLVMRegs(X85DwarfFlavour0Dwarf2L, X85DwarfFlavour0Dwarf2LSize, false);
    break;
  }
  switch (EHFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    RI->mapDwarfRegsToLLVMRegs(X85EHFlavour0Dwarf2L, X85EHFlavour0Dwarf2LSize, true);
    break;
  }
  switch (DwarfFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    RI->mapLLVMRegsToDwarfRegs(X85DwarfFlavour0L2Dwarf, X85DwarfFlavour0L2DwarfSize, false);
    break;
  }
  switch (EHFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    RI->mapLLVMRegsToDwarfRegs(X85EHFlavour0L2Dwarf, X85EHFlavour0L2DwarfSize, true);
    break;
  }
}

} // end namespace llvm

#endif // GET_REGINFO_MC_DESC

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Register Information Header Fragment                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_REGINFO_HEADER
#undef GET_REGINFO_HEADER

#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {

class X85FrameLowering;

struct X85GenRegisterInfo : public TargetRegisterInfo {
  explicit X85GenRegisterInfo(unsigned RA, unsigned D = 0, unsigned E = 0,
      unsigned PC = 0, unsigned HwMode = 0);
  unsigned composeSubRegIndicesImpl(unsigned, unsigned) const override;
  LaneBitmask composeSubRegIndexLaneMaskImpl(unsigned, LaneBitmask) const override;
  LaneBitmask reverseComposeSubRegIndexLaneMaskImpl(unsigned, LaneBitmask) const override;
  const TargetRegisterClass *getSubClassWithSubReg(const TargetRegisterClass*, unsigned) const override;
  const RegClassWeight &getRegClassWeight(const TargetRegisterClass *RC) const override;
  unsigned getRegUnitWeight(unsigned RegUnit) const override;
  unsigned getNumRegPressureSets() const override;
  const char *getRegPressureSetName(unsigned Idx) const override;
  unsigned getRegPressureSetLimit(const MachineFunction &MF, unsigned Idx) const override;
  const int *getRegClassPressureSets(const TargetRegisterClass *RC) const override;
  const int *getRegUnitPressureSets(unsigned RegUnit) const override;
  ArrayRef<const char *> getRegMaskNames() const override;
  ArrayRef<const uint32_t *> getRegMasks() const override;
  /// Devirtualized TargetFrameLowering.
  static const X85FrameLowering *getFrameLowering(
      const MachineFunction &MF);
};

namespace X85 { // Register classes
  extern const TargetRegisterClass LowRegsRegClass;
  extern const TargetRegisterClass HighRegsRegClass;
  extern const TargetRegisterClass LargeRegsRegClass;
} // end namespace X85

} // end namespace llvm

#endif // GET_REGINFO_HEADER

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Target Register and Register Classes Information                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_REGINFO_TARGET_DESC
#undef GET_REGINFO_TARGET_DESC

namespace llvm {

extern const MCRegisterClass X85MCRegisterClasses[];

static const MVT::SimpleValueType VTLists[] = {
  /* 0 */ MVT::i8, MVT::Other,
  /* 2 */ MVT::i16, MVT::Other,
};

static const char *const SubRegIndexNameTable[] = { "anonymous_2225", "anonymous_2226", "" };


static const LaneBitmask SubRegIndexLaneMaskTable[] = {
  LaneBitmask::getAll(),
  LaneBitmask(0x00000001), // anonymous_2225
  LaneBitmask(0x00000002), // anonymous_2226
 };



static const TargetRegisterInfo::RegClassInfo RegClassInfos[] = {
  // Mode = 0 (Default)
  { 8, 8, 8, VTLists+0 },    // LowRegs
  { 8, 8, 8, VTLists+0 },    // HighRegs
  { 16, 16, 8, VTLists+2 },    // LargeRegs
};

static const TargetRegisterClass *const NullRegClasses[] = { nullptr };

static const uint32_t LowRegsSubClassMask[] = {
  0x00000001, 
  0x00000004, // anonymous_2226
};

static const uint32_t HighRegsSubClassMask[] = {
  0x00000002, 
  0x00000004, // anonymous_2225
};

static const uint32_t LargeRegsSubClassMask[] = {
  0x00000004, 
};

static const uint16_t SuperRegIdxSeqs[] = {
  /* 0 */ 1, 0,
  /* 2 */ 2, 0,
};


namespace X85 {   // Register class instances
  extern const TargetRegisterClass LowRegsRegClass = {
    &X85MCRegisterClasses[LowRegsRegClassID],
    LowRegsSubClassMask,
    SuperRegIdxSeqs + 2,
    LaneBitmask(0x00000001),
    0,
    false, /* HasDisjunctSubRegs */
    false, /* CoveredBySubRegs */
    NullRegClasses,
    nullptr
  };

  extern const TargetRegisterClass HighRegsRegClass = {
    &X85MCRegisterClasses[HighRegsRegClassID],
    HighRegsSubClassMask,
    SuperRegIdxSeqs + 0,
    LaneBitmask(0x00000001),
    0,
    false, /* HasDisjunctSubRegs */
    false, /* CoveredBySubRegs */
    NullRegClasses,
    nullptr
  };

  extern const TargetRegisterClass LargeRegsRegClass = {
    &X85MCRegisterClasses[LargeRegsRegClassID],
    LargeRegsSubClassMask,
    SuperRegIdxSeqs + 1,
    LaneBitmask(0x00000003),
    0,
    true, /* HasDisjunctSubRegs */
    false, /* CoveredBySubRegs */
    NullRegClasses,
    nullptr
  };

} // end namespace X85

namespace {
  const TargetRegisterClass* const RegisterClasses[] = {
    &X85::LowRegsRegClass,
    &X85::HighRegsRegClass,
    &X85::LargeRegsRegClass,
  };
} // end anonymous namespace

static const TargetRegisterInfoDesc X85RegInfoDesc[] = { // Extra Descriptors
  { 0, false },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
  { 0, true },
};
unsigned X85GenRegisterInfo::composeSubRegIndicesImpl(unsigned IdxA, unsigned IdxB) const {
  static const uint8_t Rows[1][2] = {
    { 0, 0, },
  };

  --IdxA; assert(IdxA < 2);
  --IdxB; assert(IdxB < 2);
  return Rows[0][IdxB];
}

  struct MaskRolOp {
    LaneBitmask Mask;
    uint8_t  RotateLeft;
  };
  static const MaskRolOp LaneMaskComposeSequences[] = {
    { LaneBitmask(0xFFFFFFFF),  0 }, { LaneBitmask::getNone(), 0 },   // Sequence 0
    { LaneBitmask(0xFFFFFFFF),  1 }, { LaneBitmask::getNone(), 0 }  // Sequence 2
  };
  static const MaskRolOp *const CompositeSequences[] = {
    &LaneMaskComposeSequences[0], // to anonymous_2225
    &LaneMaskComposeSequences[2] // to anonymous_2226
  };

LaneBitmask X85GenRegisterInfo::composeSubRegIndexLaneMaskImpl(unsigned IdxA, LaneBitmask LaneMask) const {
  --IdxA; assert(IdxA < 2 && "Subregister index out of bounds");
  LaneBitmask Result;
  for (const MaskRolOp *Ops = CompositeSequences[IdxA]; Ops->Mask.any(); ++Ops) {
    LaneBitmask::Type M = LaneMask.getAsInteger() & Ops->Mask.getAsInteger();
    if (unsigned S = Ops->RotateLeft)
      Result |= LaneBitmask((M << S) | (M >> (LaneBitmask::BitWidth - S)));
    else
      Result |= LaneBitmask(M);
  }
  return Result;
}

LaneBitmask X85GenRegisterInfo::reverseComposeSubRegIndexLaneMaskImpl(unsigned IdxA,  LaneBitmask LaneMask) const {
  LaneMask &= getSubRegIndexLaneMask(IdxA);
  --IdxA; assert(IdxA < 2 && "Subregister index out of bounds");
  LaneBitmask Result;
  for (const MaskRolOp *Ops = CompositeSequences[IdxA]; Ops->Mask.any(); ++Ops) {
    LaneBitmask::Type M = LaneMask.getAsInteger();
    if (unsigned S = Ops->RotateLeft)
      Result |= LaneBitmask((M >> S) | (M << (LaneBitmask::BitWidth - S)));
    else
      Result |= LaneBitmask(M);
  }
  return Result;
}

const TargetRegisterClass *X85GenRegisterInfo::getSubClassWithSubReg(const TargetRegisterClass *RC, unsigned Idx) const {
  static const uint8_t Table[3][2] = {
    {	// LowRegs
      0,	// anonymous_2225
      0,	// anonymous_2226
    },
    {	// HighRegs
      0,	// anonymous_2225
      0,	// anonymous_2226
    },
    {	// LargeRegs
      3,	// anonymous_2225 -> LargeRegs
      3,	// anonymous_2226 -> LargeRegs
    },
  };
  assert(RC && "Missing regclass");
  if (!Idx) return RC;
  --Idx;
  assert(Idx < 2 && "Bad subreg");
  unsigned TV = Table[RC->getID()][Idx];
  return TV ? getRegClass(TV - 1) : nullptr;
}

/// Get the weight in units of pressure for this register class.
const RegClassWeight &X85GenRegisterInfo::
getRegClassWeight(const TargetRegisterClass *RC) const {
  static const RegClassWeight RCWeightTable[] = {
    {1, 8},  	// LowRegs
    {1, 3},  	// HighRegs
    {2, 6},  	// LargeRegs
  };
  return RCWeightTable[RC->getID()];
}

/// Get the weight in units of pressure for this register unit.
unsigned X85GenRegisterInfo::
getRegUnitWeight(unsigned RegUnit) const {
  assert(RegUnit < 11 && "invalid register unit");
  // All register units have unit weight.
  return 1;
}


// Get the number of dimensions of register pressure.
unsigned X85GenRegisterInfo::getNumRegPressureSets() const {
  return 4;
}

// Get the name of this register unit pressure set.
const char *X85GenRegisterInfo::
getRegPressureSetName(unsigned Idx) const {
  static const char *const PressureNameTable[] = {
    "HighRegs",
    "LargeRegs",
    "LowRegs",
    "LowRegs+LargeRegs",
  };
  return PressureNameTable[Idx];
}

// Get the register unit pressure limit for this dimension.
// This limit must be adjusted dynamically for reserved registers.
unsigned X85GenRegisterInfo::
getRegPressureSetLimit(const MachineFunction &MF, unsigned Idx) const {
  static const uint8_t PressureLimitTable[] = {
    3,  	// 0: HighRegs
    6,  	// 1: LargeRegs
    8,  	// 2: LowRegs
    11,  	// 3: LowRegs+LargeRegs
  };
  return PressureLimitTable[Idx];
}

/// Table of pressure sets per register class or unit.
static const int RCSetsTable[] = {
  /* 0 */ 0, 1, 3, -1,
  /* 4 */ 1, 2, 3, -1,
};

/// Get the dimensions of register pressure impacted by this register class.
/// Returns a -1 terminated array of pressure set IDs
const int* X85GenRegisterInfo::
getRegClassPressureSets(const TargetRegisterClass *RC) const {
  static const uint8_t RCSetStartTable[] = {
    5,0,1,};
  return &RCSetsTable[RCSetStartTable[RC->getID()]];
}

/// Get the dimensions of register pressure impacted by this register unit.
/// Returns a -1 terminated array of pressure set IDs
const int* X85GenRegisterInfo::
getRegUnitPressureSets(unsigned RegUnit) const {
  assert(RegUnit < 11 && "invalid register unit");
  static const uint8_t RUSetStartTable[] = {
    5,5,5,5,5,0,4,0,4,0,4,};
  return &RCSetsTable[RUSetStartTable[RegUnit]];
}

extern const MCRegisterDesc X85RegDesc[];
extern const MCPhysReg X85RegDiffLists[];
extern const LaneBitmask X85LaneMaskLists[];
extern const char X85RegStrings[];
extern const char X85RegClassStrings[];
extern const MCPhysReg X85RegUnitRoots[][2];
extern const uint16_t X85SubRegIdxLists[];
extern const MCRegisterInfo::SubRegCoveredBits X85SubRegIdxRanges[];
extern const uint16_t X85RegEncodingTable[];
// X85 Dwarf<->LLVM register mappings.
extern const MCRegisterInfo::DwarfLLVMRegPair X85DwarfFlavour0Dwarf2L[];
extern const unsigned X85DwarfFlavour0Dwarf2LSize;

extern const MCRegisterInfo::DwarfLLVMRegPair X85EHFlavour0Dwarf2L[];
extern const unsigned X85EHFlavour0Dwarf2LSize;

extern const MCRegisterInfo::DwarfLLVMRegPair X85DwarfFlavour0L2Dwarf[];
extern const unsigned X85DwarfFlavour0L2DwarfSize;

extern const MCRegisterInfo::DwarfLLVMRegPair X85EHFlavour0L2Dwarf[];
extern const unsigned X85EHFlavour0L2DwarfSize;

X85GenRegisterInfo::
X85GenRegisterInfo(unsigned RA, unsigned DwarfFlavour, unsigned EHFlavour,
      unsigned PC, unsigned HwMode)
  : TargetRegisterInfo(X85RegInfoDesc, RegisterClasses, RegisterClasses+3,
             SubRegIndexNameTable, SubRegIndexLaneMaskTable,
             LaneBitmask(0xFFFFFFFC), RegClassInfos, HwMode) {
  InitMCRegisterInfo(X85RegDesc, 15, RA, PC,
                     X85MCRegisterClasses, 3,
                     X85RegUnitRoots,
                     11,
                     X85RegDiffLists,
                     X85LaneMaskLists,
                     X85RegStrings,
                     X85RegClassStrings,
                     X85SubRegIdxLists,
                     3,
                     X85SubRegIdxRanges,
                     X85RegEncodingTable);

  switch (DwarfFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    mapDwarfRegsToLLVMRegs(X85DwarfFlavour0Dwarf2L, X85DwarfFlavour0Dwarf2LSize, false);
    break;
  }
  switch (EHFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    mapDwarfRegsToLLVMRegs(X85EHFlavour0Dwarf2L, X85EHFlavour0Dwarf2LSize, true);
    break;
  }
  switch (DwarfFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    mapLLVMRegsToDwarfRegs(X85DwarfFlavour0L2Dwarf, X85DwarfFlavour0L2DwarfSize, false);
    break;
  }
  switch (EHFlavour) {
  default:
    llvm_unreachable("Unknown DWARF flavour");
  case 0:
    mapLLVMRegsToDwarfRegs(X85EHFlavour0L2Dwarf, X85EHFlavour0L2DwarfSize, true);
    break;
  }
}



ArrayRef<const uint32_t *> X85GenRegisterInfo::getRegMasks() const {
  return None;
}

ArrayRef<const char *> X85GenRegisterInfo::getRegMaskNames() const {
  return None;
}

const X85FrameLowering *
X85GenRegisterInfo::getFrameLowering(const MachineFunction &MF) {
  return static_cast<const X85FrameLowering *>(
      MF.getSubtarget().getFrameLowering());
}

} // end namespace llvm

#endif // GET_REGINFO_TARGET_DESC

