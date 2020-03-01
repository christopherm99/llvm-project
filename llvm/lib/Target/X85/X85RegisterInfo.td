// General Purpose Registers
// 8-bit Registers

class X85Reg<string n> : Register<n> {
  let Namespace = "X85";
}

class R8<string n> : X85Reg<n>;

class R16<string n, list<Register> subregs> : X85Reg<n> {
  let SubRegs = subregs;
  let SubRegIndices = [SubRegIndex<8>, SubRegIndex<8,8>]; // Taken from SparcRegisterInfo.td
}

def A : R8<"A">, DwarfRegNum<[0]>;
def B : R8<"B">, DwarfRegNum<[1]>;
def C : R8<"C">, DwarfRegNum<[2]>;
def D : R8<"D">, DwarfRegNum<[3]>;
def E : R8<"E">, DwarfRegNum<[4]>;
def H : R8<"H">, DwarfRegNum<[5]>;
def L : R8<"L">, DwarfRegNum<[6]>;

// Special 8-bit
def SPL : R8<"SPL">, DwarfRegNum<[7]>;
def SPH : R8<"SPH">, DwarfRegNum<[8]>;
def PCL : R8<"PCL">, DwarfRegNum<[9]>;
def PCH : R8<"PCH">, DwarfRegNum<[10]>;

// Joint 16-bit
def HL : R16<"HL", [H, L]>, DwarfRegNum<[11]>;
def SP : R16<"SP", [SPH, SPL]>, DwarfRegNum<[12]>;
def PC : R16<"PC", [PCH, PCL]>, DwarfRegNum<[13]>;

def LowRegs : RegisterClass<"X85", [i8], 8,
  (add A, B, C, D, E, L, SPL, PCL)>;

def HighRegs : RegisterClass<"X85", [i8], 8,
  (add H, SPH, PCH)>;

def LargeRegs : RegisterClass<"X85", [i16], 8,
  (add HL, SP, PC)>;
