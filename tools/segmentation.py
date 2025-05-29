import re

# NOT FINISHED 
# NEED TO TEST SOME EDGE CASES


M = "ျ  ြ  ွ  ှ" #medials
C = "က-အ" #consontants
V = "ါ  ာ  ိ  ီ  ု  ူ  ‌ေ  ဲ " #Vowels
S = "္" #ပက်ဆင့်
AtTat = "်" #အသက် 
F = "့  ံ  း"  #final additional signs
E = "ဣဥဦဩ၎"   #additional vowels
I = "ဤဧဪ၌၍၏" #idependent vowels
N = "ဿ" # ဿ
allEngish = "A-Za-z0-9" #all English alphabets

punctuation = "၊။" #punctuation
digit = "၀-၉" #digits


# Remove whitespace
M = "".join(M.split())  
V = "".join(V.split()) 
F = "".join(F.split())


# NOTE: Myanmar text usually has 0(zero) and ဝ (wa) interchangeably uncommend bottom line for that usecase
#C = "က-အ၀" #consontants (including 0(zero))


# TODO: for S = "္" #ပက်ဆင့် some additional rules are needed??
pattern = (
    fr"[{C}][{M}]*[{V}]*[{F}]?[{C}]{AtTat}[{F}]?"        # Consonant clusters with optional final
    fr"|[{C}][{M}]*[{V}]*[{F}]?[{C}][{F}]{AtTat}"        # final before Atat to accomndate user input misalignmnet
    fr"|[{C}][{M}]*[{V}]*[{C}][{S}][{C}][{M}]*[{V}]*[{F}]*{AtTat}?" # To accomodate pat sin
    fr"|[{C}][{M}]*[{V}]*{AtTat}"            # With Asat
    fr"|[{C}][{M}]*[{V}]*[{F}]*"              # Basic syllables
    fr"|[{E}][{C}]{AtTat}[{F}]?"                    # Extended vowels with consonant
    fr"|[{I}]"                               # Independent vowels
    fr"|[{digit}]+"                           # Digits
    fr"|[{punctuation}]"                     # Punctuation
    fr"|[{N}]"                               # Special consonant
    fr"|[{allEngish}]+"                             # A to Z
)


def segment_characters(text):
    return re.findall(pattern, text)
