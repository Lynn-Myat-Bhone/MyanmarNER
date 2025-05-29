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


test_string = """English ဘာသာ Lorem ipsum စာများသည် ယေဘုယျအားဖြင့် pseudo-Latin စကားလုံးများဖြင့် ဖွဲ့စည်းထားခြင်းဖြစ်ပါသည်။ ဒီဇိုင်းများကို နမူနာပြသရန် နှင့် ဒီဇိုင်၏ အရည်အသွေးကို စမ်းစစ်ရန် အတွက် placeholder text အနေဖြင့် ကျယ်ပြန့်စွာ အသုံပြုလျှက်ရှိပါသည်။ စကားလုံးများတွင် အဓိပ္ပာယ် မရှိသည့်အားလျှောစွာ ဖတ်ရှုသူ အနေနဲ့ စာကို ဖတ်ရှုနေစရာ မလိုအပ်ပဲ ဒီဇိုင်းကို ပိုမို အာရုံစိုက်နိုင်ပါသည်။ ဘာသာစကား အသီးသီးတွင် Lorem ipsum များရှိသကဲ့သိုပင် ယခုမှာ မြန်မာ ဘာသာ အတွက် ဖြစ်ပါသည်။

ယခု စာမျက်နှာ သည် Lorem ipsum များကို မြန်မာ Version အနေနဲ့ ထုတ်ပေးထားခြင်းဖြစ်ပါသည်။ စာပိုဒ် (၅) ပိုဒ်ပါ၀င်ပြီး စာပိုဒ် တစ်ခုချင်းစီတွင် ၀ါကျ (၅) ခုမှ (၆) အထိပါ၀င်ပါသည်။ စာလုံးတိုင်းတွင် လူသုံးနည်းသော ပါဠိ စာတစ်၀က် နှင့် လူသုံးများသော မြန်မာစာ တစ်၀က် ပါ၀င် ပါသည်။ ပါ၀င်သော ပါဠိစာများသည် ပုံမှန် စာဖတ်သူများ အတွက် ဖတ်ရှုရန် ခက်ခဲသော စာများဖြစ်စေပါသည်။ pseudo-sentence စာများကို generate လုပ်သည့် အဆင့် တစ်ဆင့်စီကို အောက်ပါစာပိုဒ်တွင် ဖော်ပြထားပါသည်။

ပါဠိစာအမှန် (၁၂) ကြောင်း နှင့် မြန်မာ စာအမှန် (၁၂) ကြောင်း၏ ဝဏ္ဏ syllable များကို ကျပန်း ရောမွှေ ပါသည်။ အဆိုပါ ကျပန်း ဝဏ္ဏ (၂) လုံး မှ (၆) လုံးအထိကိုစုစည်းပြီး ကျပန်း ပုဒ်စု phrase ကိုရရှိပါသည်။ အဆိုပါ ပုဒ်စု (၃) ခုမှ (၉) ခုကို စုစည်းပြီး ၀ါကျ sentence များကို တည်ဆောက်ပါသည်။ ရလဒ်အနေနှင့် စာအမှန်နှင့် အမြင်တွင်လွန်စွာ ဆင်တူပြီး ဖတ်ရှုရန် မလွယ်သော ၀ါကျများကို ရရှိပါသည်။"""



with open("output.txt", "w", encoding="utf-8") as f:
    result = segment_characters("ကိစ္စ ၀တ္ထု")
    print(result, file=f)


# some weird rule for text like => မင်္ဂလာပါ / မားစ်ဂြိုလ်
# working normally for other cases