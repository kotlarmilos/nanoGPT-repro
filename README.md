# NanoGPT-repro

A minimal repro of ChatGPT based on https://github.com/karpathy/nanoGPT.

This repo demonstrates how to build, train, and sample from a small transformer-based language model with multi-block attention.

## Features

- Token & positional embeddings
- Multihead transformer blocks
- Simple feed-forward residual blocks with normalization
- Multinomial sampling

## Requirements

- Python 3.8+
- PyTorch 1.12+

Install dependencies:

```bash
pip install torch
```

## Hyperparameters

| Name            | Default | Description                                   |
| --------------- | ------- | --------------------------------------------- |
| `batch_size`    | 32      | Parallel sequences per iteration              |
| `context_size`  | 256     | Maximum context length                        |
| `n_embed`       | 512     | Embedding dimension                           |
| `n_transformers`| 16      | Number of transformer blocks                  |
| `n_iters`       | 15000   | Total training iterations                     |
| `learning_rate` | 1e-3    | Adam optimizer learning rate                  |
| `log_interval`  | 100     | Steps between training logs and sample outputs|

## TODOs
- [ ] Replace char-level tokenization with `tiktoken` / `sentencepiece`
- [ ] Add model checkpointing & resume capability
- [ ] Implement learning rate scheduler
- [ ] Support multi-GPU / distributed training
- [ ] Reduce precision
- [ ] Load GPT-2 weights and run benchmarks

## Training example

```
nO.W.lBsqNtedZoAx3s&i-?svfgtItQtyAggL
WxV33C$mNeRkEMRqGaJXAUeCEw!.UeqsiSF$-yVidGPriE;yo:TraHZMvU?kq&
[100/15000] loss=2.5774 time=4.40s

TENAP Ge in.

Wy

Laresuncon:
B ayooour dsun veay
BERENIAN:
Hur I d' aneronk, Tanr do, thant whana p
[200/15000] loss=2.5213 time=3.09s

ARIO:
T:
HA:
GTBENA:
BEPERI'd l,
BESTI IUSonf arsher, lorr, hervee y t drifreld oran; wndrftePRDI fo
[300/15000] loss=2.5395 time=3.49s

KI Giead;
J ch Scurlyokn I I lary offruldeleay blivend r, d harouefl yllen d nt ou tus t D o t iomip
[400/15000] loss=2.4952 time=3.10s

NRIS:
CENumbrag 'sous, on g st, y,
AR:
Tethid nu, oe tor, maghisto ike sheflunen bepin sie h vved Bu
[500/15000] loss=2.5130 time=3.10s

MINolifll w EMuioshe ospen mey anes And t tarthupulin hergavelixPSwin,
Is.
Wis mut,
MARY:
hedare d t
[600/15000] loss=2.4584 time=3.50s

JULIs shid enok mayor t afouresedeserovil m; we l then, hencexoly in bo t ms nd wennatheve, andemer 
[700/15000] loss=2.4693 time=3.11s

Yornto'r' bo an fud mardag nicthimfeisth, hee
Whimy suks ht t md s,
I te hind.
Hary n, ere y?
Theare
[800/15000] loss=2.4720 time=3.11s

BRDE:
ASSorit ustoef ante, youhe, fen a onghe lecungad br atrtld sketh per ato'e nt
An t
Tist d llie
[900/15000] loss=2.4546 time=3.43s

OR:
ere, shese thereors lipodee thainerst fllor cat bounghst:
G weatthancusprbrevito wr s I thecheve
[1000/15000] loss=2.4426 time=3.13s

Thisurll Sicigecl ta frrone om yopre he d anin drd:
Anamncesby nd wancau profopo het oue mofowepll t
[1100/15000] loss=2.4495 time=3.10s

llsar; hethend mothite ne 'denk lan cut , ungune youctadve ofen wo,
TENo ht th to f ily wipe tree we
[1200/15000] loss=2.4193 time=3.31s

The rwald, omy lilughaly mat at nnececo intestinuche
Al; tasherr.
tid muriut s, weerig lan
ingle pt.
[1300/15000] loss=2.4417 time=3.09s

Son hay, tounou sind male vee juplave pior vemih fine rthoangundtin bllt t se fans losees,
Ipro Bsan
[1400/15000] loss=2.3619 time=3.27s

PETIU, ROPRIO:
Theat toor I:
The fut no thicknge th ' boysis ing, amance ald
YBy
I er, berok, acty m
[1500/15000] loss=2.3666 time=3.09s

PONGLETIO:
Whargheind too, chas sie sthand satat,
Cichefar aty?
Nerste lene bas hiofing to fe, goupl
[1600/15000] loss=2.3662 time=3.09s

Sor orkse piat's you.

RUKELINGRUMED:
WI.w,-noHEET:

I meeereeser thy th ay, ERl n, relvyous ang we 
[1700/15000] loss=2.3651 time=3.52s

Le co thar; bof ming ot of ceme Yousut,
To s, tho wich
Dur and, ase inf; qum sharpaken,
A saf vethat
[1800/15000] loss=2.3508 time=3.08s

PERET:
By cewis Youe, ior, geadamee.
TRICHORK:
OR HASD Her brasteaver den ell 'suthaes f hach b to m
[1900/15000] loss=2.3314 time=3.08s

ARIIENETNA:
Ad you sighes mioattaby, tan tull I'sk tid in l wid thel de ff ho'd bed. veat--be ce
I s
[2000/15000] loss=2.3253 time=3.48s

DINIUS:
Sh, you if kive frat fin cop oneseven.

ABRA:
Horwetie yot aru toles, ghech, Ther.

DUCHese 
[2100/15000] loss=2.2765 time=3.09s

LUCYOSWhy Have vegror IORKI Viere tus.

Sit to poht Yer Now Eecand wout hof thie fees,
An no rerestr
[2200/15000] loss=2.2969 time=3.10s

HOREN:
Wher at lo Whe lim ly not hithrif harenserw,
Bup lordigne wh nosivend therithat Cowedioun' he
[2300/15000] loss=2.2463 time=3.50s

LYCVOKHARG:
O, low doknd' that orvest uncou;
E poak a for d her tof wer worer
Jolebr:
soretthe; bior
[2400/15000] loss=2.2702 time=3.11s

CADUKA:
Iapper:
He min, atem hes, tou fing ave thindey xwost my kee;
Andy's.
Pher:

Didbomy, le bu t
[2500/15000] loss=2.2439 time=3.10s

ApcIA:
I forve woold, servar ampith lof Wer, aind oon batce?
-vele Il On; sear ghe thentapacor of wh
[2600/15000] loss=2.2322 time=3.38s

Nook hise or irgld you u, thiske theres
Mazen rurl, top youreppis thencate,
Alend ir furene fued you
[2700/15000] loss=2.2071 time=3.10s

Whe Ind ma dose, wheay; the mend.

If linis mis ye s it oong ofom?

MERINIO:
If wiw tho say uerk ind
[2800/15000] loss=2.2028 time=3.21s

COMPRO MIO:
dy ofever hat nawdourshis sand.


Reapromy lf cay youed te mouthewlat of mit Mall yon's 
[2900/15000] loss=2.2530 time=3.12s

BRUSIIA:
No fring mameapdey ing inevilk keeth
Thess and whis kinterf bin bre.

HUS:
Thour my de ro h
[3000/15000] loss=2.2052 time=3.11s

whis cay ired, Carithe ton: of hou whena flite siss shandother
ch an te mumin joroastoled a; in; tod
[3100/15000] loss=2.1912 time=3.38s

FLYUTou. ANEL:
My, hy I not urt, tur four
Trourser tirevent in at aitheruen!
Sh me.

Ye BUMERLYerl I
[3200/15000] loss=2.2174 time=3.09s

IASABELA:
The bruneptie ing im:
Letht blonis, tuiger weat grounth dear sat hoir pio senught
CAPTyir 
[3300/15000] loss=2.1870 time=3.10s

Shay for Of weds shimhedned ly chiss erdes med----nare!

INCOLIS:
LARI be crinals pherem Sails ayeas
[3400/15000] loss=2.1582 time=3.50s

Hor pot's and nan sed whathree,'le, she las,
I wat, cove, is halagove oftief?

LANUS:
Therkeis, woul
[3500/15000] loss=2.1845 time=3.09s

Less EDY lie bris siding to is mirncompend haywe: ing dove boonglo now wete sight
Do your:
Wirsem Do
[3600/15000] loss=2.1578 time=3.09s

Andses lell of donimy, don, will dam rrambe id wit
HOMy lake sen, is mourten thesiny!

LUCIO:
I'Thad
[3700/15000] loss=2.1159 time=3.50s

Peaviog atits all maslet, gove to? tanak!
Ou mee ing, meno wont ustinencutnus; far thy wawe se bat R
[3800/15000] loss=2.1350 time=3.11s

Anos betar whous wer angeseered eiff the the, menest ne
my alie'es the, tot eruin at me of aris and 
[3900/15000] loss=2.1239 time=3.09s

Whis POLentas, kirenque if som rof the?
I wit. What morince thy my ou oer?
To Lust ort:
I the it ate
[4000/15000] loss=2.1474 time=3.38s

ERMISABlA OFLUCINI:
Wonthes alame!
OLARTHARD the yous.

VASLENO:
ANA:
Plas, befo, not coohe to spere
[4100/15000] loss=2.1087 time=3.09s

OMEY:
Whe hour porsuste your predy: kelinest, itl bed; of
Andeat and off rovest 'st thid lois maneas
[4200/15000] loss=2.1467 time=3.14s

Eds inged'd you an exave idve mpprose hand-mast
To wuld irn, to sone tay, thy bu ard whith youre.

G
[4300/15000] loss=2.1414 time=3.22s

Ano come jutintest den her!
Heas comall bufath abe to with then:
Nor thos hous of stiss not theave c
[4400/15000] loss=2.1073 time=3.10s

Loreh hath wo thouns wer dyse med aver hervagey:

thy all that sir quou woy on oungar fall'tver hant
[4500/15000] loss=2.0983 time=3.31s

I RABETRUSICKE VINCENTENRY:
He the sarrod gone when say: att I wards titaut your weed dome
Pry, prau
[4600/15000] loss=2.0849 time=3.09s

Home ffora ad of sulers worder witthy lowe.

MORIOLO:
De asto.

QUEETHIZELONERIZARGHETH:
Nos, now th
[4700/15000] loss=2.0871 time=3.09s

I my sha plloffring to keatl wis th truchilcine.

MIRICHARINAND: hey ford, '
Whim'd If, I ke fiellI 
[4800/15000] loss=2.1168 time=3.48s

Led onot new mais; I have st ay mon,
Wello sefort ermed? st nowhin Sith crs I me for
WARLY:
Whel mas
[4900/15000] loss=2.0830 time=3.09s

Tre My Cristesty wary's, wherseto-se
Buthich, you kn thourt corsucteree, for Whoukbe halent of.
Bich
[5000/15000] loss=2.1142 time=3.09s

TRUCENIUS:
O, I makine ss; An for im, I gaas ay'gur puld de steks seat for siolce
TOR: are be hare y
[5100/15000] loss=2.0795 time=3.53s

CLAnd nor:
Nahee no you sur firecence of ve louse and!

KING EDWI:
Your the you bot edrcrow the, the
[5200/15000] loss=2.0899 time=3.11s

IA I am pinshe kist hice rvate? you 's her,
Ty torme you des any bet upoing forey the hon citty od y
[5300/15000] loss=2.0681 time=3.09s

And brancper nou this solk Joach to Cleand Gord;
Ingm ood er to whee's to book not heed:
Clmeand abl
[5400/15000] loss=2.0429 time=3.47s

KING RIA:
Selast warlith Rell bel ato dupen:
Shong, fabe grut fess wire bento may.

Segelved 'twitt 
[5500/15000] loss=2.0065 time=3.09s

OMIONCANTIUS:
Sheret thy o'lle Get rome how,
I glane nie be do the will, seribest. Roxe
WARWER:
MaNo
[5600/15000] loss=2.0497 time=3.09s

Fill gant to vor fe eces ate thyou flith, the mave
fres mono to momponmeswe I sell!
To ange youmat d
[5700/15000] loss=2.0520 time=3.31s

Edwouke weve and this of to my Live thyet,
Men rehe han lat, the but her citte
wis o'lets amaith cor
[5800/15000] loss=2.0532 time=3.09s

RIANA:
In by tood, sppllese deyese.
SICHARPTIUR:
Clt, did's yous peap!.

NENES:
Manto frige, ing ty 
[5900/15000] loss=1.9951 time=3.21s

Myer Marry  nourrive, ace of wifly you ware! hrit, i's arane;
And wit sond loo Govancoul, Andr:
When
[6000/15000] loss=2.0569 time=3.09s

TARINIUS:
Wer?, having you.

not you mone by Mare of tene my sor yous:
lony you den! Wee a parmeksed
[6100/15000] loss=2.0481 time=3.10s

AUFFORT:
But ro be as cyaling in thie onson's, to so my that,
Ausee? staraint, thy beeppeak end;
Who
[6200/15000] loss=2.0574 time=3.38s

And Not Trized, intry kifes son a cince?

LY mase nouch mme: a butt, ando a turme
not lelf as and ou
[6300/15000] loss=1.9909 time=3.09s

Af LA.

SINGYBOLBEREY:
Ond you, ne bertwem, by this dever
Aysbblotnsseone, le decefard thisw thee kn
[6400/15000] loss=2.0362 time=3.10s

Thmigh undeamof cam ortie, to mor tare aen.

CATIONDY, I weos:
That mentinges cavan: putens you, had
[6500/15000] loss=2.0392 time=3.50s

He no you juccend you snins itiese fear the ey
Meney, anchime? I king, I is and rur ant-
My shanat a
[6600/15000] loss=2.0092 time=3.09s

Gor mornot be Edwich himest bout int amas
Welt Beawnft le then iss do anated eever the
A's cusas uff
[6700/15000] loss=1.9917 time=3.10s

Citis.

Claurshin hurguown, wherband grand do shat mightle ssong.

FLACUS: Lerment, Ale!
My have had
[6800/15000] loss=1.9760 time=3.49s

SuENV:
Yours thare tis; be chor, yere?-

Nor Til Hord be ande to hing andstir? Ox
Blownel sseave ipp
[6900/15000] loss=2.0084 time=3.09s

ENO: abrotwe stas woon exceout,
WARWAll der leab I by hostime dof hicain bre!'s 'twerefough boon
Fir
[7000/15000] loss=1.9682 time=3.10s

Reeer main tors, onothes dove the, kear, by dels fato cure
O him so the now makenvin sailmes
Grod; F
[7100/15000] loss=1.9578 time=3.42s

Wher mereve gan.

KING RE:
ENCE:
AllIO fear KUCHARD IV:
I make. Caray, I scon thoushe man.

Gork IMB
[7200/15000] loss=2.0257 time=3.13s

Butefl olock offiere as lowe to thring outws drown
Wemat's thalt:
Ay filure to sow one is marnet ora
[7300/15000] loss=1.9861 time=3.17s

My counecion sin unes?
Now'sttice yeu pracy som cherlow ther.
We wich be gemenen the ur'd I my cour 
[7400/15000] loss=1.9952 time=3.17s

ISABELLANUS:
Ald way, thour withes; muit your dit may ound in's.
Nay lord in freloust
By band to lac
[7500/15000] loss=1.9617 time=3.09s

ONVOLIANUS:
Sechare, ay mest coglove hour tanly?

Tour to if rome amm back thath prow; of teen,
I fo
[7600/15000] loss=1.9824 time=3.30s

Asengmint mand whell as the hee a hou usomingbee
Soun of ay beald noblesis.

IUSt Lecilus'd divesh, 
[7700/15000] loss=1.9850 time=3.09s

Bkine bancellederagoo hiing, I'll ant the kity
your-randusure is whick struct to noth me tor It.

We
[7800/15000] loss=1.9985 time=3.09s

ANGSABEL:
And guze tre, the noud what hamse;
And myst a true kne. Bus
Ayourve, my kinow sty Lame so?
[7900/15000] loss=1.9819 time=3.49s

REMION Your vose stukiny fartans?

MEO:
You eame ame:

My ought blook, lorkse yif or him seawn thing
[8000/15000] loss=1.9672 time=3.09s

The nowcarffelsces on him.

If berate?
ISICINIUS:
I Duley cavence spondurs, sael you with!

MERCHENR
[8100/15000] loss=1.9752 time=3.11s

BANTALLONT:
Now ite:
Is that nou sunce ace: love with ud kife.

JULIET:
Face be whicot you feet loge
[8200/15000] loss=1.9500 time=3.48s

MIS ICIALLA:I liss!

DUKEHNIO: barts, sir; be dong noke, the lew, the  bred
Whend evese a him to gfo
[8300/15000] loss=1.9392 time=3.09s

mecsance say fire of thou the Dime quaseself.
The is my dith bles, ge futhe lives him have ange
Ytud
[8400/15000] loss=1.9486 time=3.10s

Edmontur loble, mold bigh an but in Galchinced;
Whembur daddion them you his is blovard, geats.
Yert
[8500/15000] loss=1.9721 time=3.43s

WARANTY Brappanty rets, wend,
And marr fall sine, not, nrese in by lusue no yof his
And of aatey be 
[8600/15000] loss=1.9437 time=3.09s

Yought chou by sour goavivar heent, no tood,
Wetcinds!
Grent! 'taks: ank unteld, mut'll prep-
My, ro
[8700/15000] loss=1.9004 time=3.09s

RCHARD ENCE:
QUEGARET:
Angemand matey ss lold Edlaneess!

ARDWARWARW:
Which endearly, youl sible now
[8800/15000] loss=1.9253 time=3.32s

JURET:
Mick ow
Dreing O fore peat mook meserve bame's thereproserves:
Ah I wnows anly come vine fir 
[8900/15000] loss=1.9600 time=3.09s

LUCIO:
To eave he why crity I herle, ente st
Goult spalines my aray.

There trath ipot betiesh they 
[9000/15000] loss=1.9541 time=3.24s

Man upond one mot uchan oove surtar carn
The gooke low me ersefuldongency.
Norse, give le have seeve
[9100/15000] loss=1.9632 time=3.09s

Leth usta theave sor de amtacess, blonceed,
And vits, for been; the sher neot.

ESCAlck, and make ar
[9200/15000] loss=1.9054 time=3.10s

EYBALLA:
Lard him. Bot theady
Hords, lif live you. Bisemes'd it beast,
And ler woited goth cantintur
[9300/15000] loss=1.9343 time=3.45s

Ait Hises for ace choughter therds to-d the in she anted,
Ald faolet sulf ons'd in that be seat you,
[9400/15000] loss=1.9186 time=3.10s

FOMILABELLOUS:
O, my IN thout seep.
Opcety compoibesen:
We thy therving is amey nars theanter.

LIZE
[9500/15000] loss=1.9372 time=3.24s

FLODE Lord Wide, dwitn phore provil is ding,
I'll a shattenemy crmis oud pad made fruse you massexfu
[9600/15000] loss=1.9558 time=3.53s

KING RIO:
Sam?

BIANTA:
'el, balty, whand beatell se ie, mastid!

KENRY VIO:
Her, anosglice orly tha
[9700/15000] loss=1.9245 time=3.11s

IAN:
Shy. I.

BOLTo sleeem?

Gmien, serve for But ind joy used.

LUCENTIO:
Nave be eade, I do re no'
[9800/15000] loss=1.9856 time=3.11s

I do sonfall queed icenver meerfateduty isgre!
Wher he mopt ishe faur that of thase.
Thhon, but lar:
[9900/15000] loss=1.9450 time=3.48s

oonse is hing on this bleor coust.

COMIO:
You, werosteed?

MENENIUS:
Ay, mor gete, is to quans oond
[10000/15000] loss=1.8674 time=3.10s
```
