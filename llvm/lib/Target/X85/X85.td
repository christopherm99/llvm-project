include "llvm/Target/Target.td"

include "X85RegisterInfo.td"
include "X85InstrInfo.td"

def X85AsmParser : AsmParser;
def X85InstrInfo : InstrInfo;


def X85 : Target {
  let InstructionSet = X85InstrInfo;
  let AssemblyParsers  = [X85AsmParser];
}