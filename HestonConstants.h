#pragma once

#define PARAMETERS_COUNT 5
#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078

// For Gauss-Legendre Quadrature

#define INTEGRATION_POINTS 96

const double Point[] = {
	1.5524805838461658615494710781724035922726865E-4,
	8.178120684091611379252802366403183062029301E-4,
	0.0020090785063953546748004575603216997412987204,
	0.00372804983811868771405385114919942956153746515,
	0.0059729368351881002596186137907744027247676997,
	0.00874136821849266127647708405872594102751209345,
	0.0120304127074317667736994828790600454007985227,
	0.01583658576836789391317031677817447080367787595,
	0.02015585427562873034996598705763262940632581675,
	0.02498364110778118212195052577806825609514049775,
	0.03031483012362239153407131028419239082236592645,
	0.03614377163884565451765476365172680654710875715,
	0.04246428843955096289705776692478922895646179055,
	0.04926968234207382934038365134515658107956431015,
	0.0565527412987897919715612833031069205840036775,
	0.06430574704535174856311261775347864263840068145,
	0.072520483282699272268606515053215369678601985,
	0.08118824438590643925284859161765357151025957845,
	0.0902998446310341622305001878877609824323263903,
	0.0998456279304295913856019280143361583917722123,
	0.1098154780662833911981977720553411259899083093,
	0.1201988294116762506485147794401891920732906578,
	0.13098467812779993357441713445175144738514014705,
	0.14216159382551618688742792597118757235121001475,
	0.15371773167891421932787711509087872027295173535,
	0.16564084497804192302372139621361290548775468955,
	0.17791829810751644660079382496834921884845039795,
	0.19053707993726571480681535650606860130703784785,
	0.20348381761121395965822212247812089341319893125,
	0.21674479071930141579787490245826868384465869725,
	0.23030594583782128188659870133721968091273468545,
	0.2441529114226661632072451272855728483752776303,
	0.258271013039701820115797195318052372264374191,
	0.27264528891612849568216192595684609349799644205,
	0.287260505796349727317590398215001027592605181,
	0.30210117508554569835749987843241792014390831215,
	0.3171515692638431824845522030022634113844062755,
	0.3323957385536872886918371875971856639064564469,
	0.34781752782275182348803511035239834092441144085,
	0.3634005937044754292563639221377176027759080023,
	0.3791284219180799938360340481296861648442208888,
	0.3949843447697163981985764071438521124278849453,
	0.41095155881619069862029868741175424254280583485,
	0.4270131426725515290054463333283367697628455042,
	0.4431520749446670395443959522928359596872626873,
	0.45935125226778722050276436851262357061085255635,
	0.47559350743197513444402089786936247145952443715,
	0.49186162757519851521043271815238077321897705495,
	0.50813837242480148478956728184761922678102294505,
	0.52440649256802486555597910213063752854047556285,
	0.54064874773221277949723563148737642938914744365,
	0.5568479250553329604556040477071640403127373127,
	0.5729868573274484709945536666716632302371544958,
	0.58904844118380930137970131258824575745719416515,
	0.6050156552302836018014235928561478875721150547,
	0.6208715780819200061639659518703138351557791112,
	0.6365994062955245707436360778622823972240919977,
	0.65218247217724817651196488964760165907558855915,
	0.6676042614463127113081628124028143360935435531,
	0.6828484307361568175154477969977365886155937245,
	0.69789882491445430164250012156758207985609168785,
	0.712739494203650272682409601784998972407394819,
	0.72735471108387150431783807404315390650200355795,
	0.741728986960298179884202804681947627735625809,
	0.7558470885773338367927548727144271516247223697,
	0.76969405416217871811340129866278031908726531455,
	0.78325520928069858420212509754173131615534130275,
	0.79651618238878604034177787752187910658680106875,
	0.80946292006273428519318464349393139869296215215,
	0.82208170189248355339920617503165078115154960205,
	0.83435915502195807697627860378638709451224531045,
	0.84628226832108578067212288490912127972704826465,
	0.85783840617448381311257207402881242764878998525,
	0.86901532187220006642558286554824855261485985295,
	0.8798011705883237493514852205598108079267093422,
	0.8901845219337166088018022279446588740100916907,
	0.9001543720695704086143980719856638416082277877,
	0.9097001553689658377694998121122390175676736097,
	0.91881175561409356074715140838234642848974042155,
	0.927479516717300727731393484946784630321398015,
	0.93569425295464825143688738224652135736159931855,
	0.9434472587012102080284387166968930794159963225,
	0.95073031765792617065961634865484341892043568985,
	0.95753571156044903710294223307521077104353820945,
	0.96385622836115434548234523634827319345289124285,
	0.96968516987637760846592868971580760917763407355,
	0.97501635889221881787804947422193174390485950225,
	0.97984414572437126965003401294236737059367418325,
	0.98416341423163210608682968322182552919632212405,
	0.9879695872925682332263005171209399545992014773,
	0.99125863178150733872352291594127405897248790655,
	0.9940270631648118997403813862092255972752323003,
	0.99627195016188131228594614885080057043846253485,
	0.9979909214936046453251995424396783002587012796,
	0.9991821879315908388620747197633596816937970699,
	0.99984475194161538341384505289218275964077273135
};

const double Weight[] = {
	7.967920655520124294381434969435687599310869E-4,
	0.0018539607889469217323359253508939105882082884,
	0.0029107318179349464084106179894007250097471693,
	0.0039645543384446866737334157674196598776912479,
	0.0050142027429275176924701949690308984740741125,
	0.0060585455042359616833167420317290879695781512,
	0.0070964707911538652691441608121433919347720993,
	0.00812687692569875921738242770785593773424476032,
	0.00914867123078338663258460266520792852644880197,
	0.0101607705350084157575876369538253688489848459,
	0.0111621020998384985912132638285620686108639634,
	0.0121516046710883196351813527366672521223706439,
	0.0131282295669615726370636669025991088987756993,
	0.014090941772314860915861624724636639518392151,
	0.015038721026994938005876275222097552837384834,
	0.0159705629025622913806164567914930510609914827,
	0.0168854798642451724504775406068619266525996141,
	0.0177825023160452608376142264860710160638992399,
	0.01866067962741146738515675862213221906414405171,
	0.019519081140145022410085220281204686487858881,
	0.0203567971543333245952452154172716143420689696,
	0.02117293989219129898767386719100370076761697637,
	0.0219666444387443491947563868015622362501689564,
	0.0227370696583293740013478419774902863557003246,
	0.02348339908592621984223593266761258095476559518,
	0.0242048417923646912822673378726770389498324916,
	0.02490063322248361028838218086833273645367805861,
	0.0255700360053493614987971679436000860963400547,
	0.0262123407356724139134579639644633435616059973,
	0.0268268667255917621980567287141566458653227208,
	0.02741296272602924282342108748909127069857296107,
	0.02797000761684833443981857658902250784489130589,
	0.02849741106508538564559951294580560456969729835,
	0.02899461415055523654267878127968157305501550092,
	0.0294610899581679059704363321828584492516225204,
	0.0298963441363283859843880757944000605401611633,
	0.0302999154208275937940887642065009140703177802,
	0.03067137612366914901422883035620427627242384548,
	0.0310103325863138374232497799706326368443931364,
	0.0313164255968613558127842667150631281757102946,
	0.03158933077072716855802074616996416012503447119,
	0.03182875889441100653475373988553225349789089889,
	0.0320344562319926632181389774702111629945035792,
	0.0322062047940302506686671145572325185039177934,
	0.0323438225685759284287748388289432704281009629,
	0.0324471637140642693640127884488458583333736353,
	0.03251611871386883598720549144777835466900356913,
	0.0325506144923631662419614182972857314873080135,
	0.0325506144923631662419614182972857314873080135,
	0.03251611871386883598720549144777835466900356913,
	0.0324471637140642693640127884488458583333736353,
	0.0323438225685759284287748388289432704281009629,
	0.0322062047940302506686671145572325185039177934,
	0.0320344562319926632181389774702111629945035792,
	0.03182875889441100653475373988553225349789089889,
	0.03158933077072716855802074616996416012503447119,
	0.0313164255968613558127842667150631281757102946,
	0.0310103325863138374232497799706326368443931364,
	0.03067137612366914901422883035620427627242384548,
	0.0302999154208275937940887642065009140703177802,
	0.0298963441363283859843880757944000605401611633,
	0.0294610899581679059704363321828584492516225204,
	0.02899461415055523654267878127968157305501550092,
	0.02849741106508538564559951294580560456969729835,
	0.02797000761684833443981857658902250784489130589,
	0.02741296272602924282342108748909127069857296107,
	0.0268268667255917621980567287141566458653227208,
	0.0262123407356724139134579639644633435616059973,
	0.0255700360053493614987971679436000860963400547,
	0.02490063322248361028838218086833273645367805861,
	0.0242048417923646912822673378726770389498324916,
	0.02348339908592621984223593266761258095476559518,
	0.0227370696583293740013478419774902863557003246,
	0.0219666444387443491947563868015622362501689564,
	0.02117293989219129898767386719100370076761697637,
	0.0203567971543333245952452154172716143420689696,
	0.019519081140145022410085220281204686487858881,
	0.01866067962741146738515675862213221906414405171,
	0.0177825023160452608376142264860710160638992399,
	0.0168854798642451724504775406068619266525996141,
	0.0159705629025622913806164567914930510609914827,
	0.015038721026994938005876275222097552837384834,
	0.014090941772314860915861624724636639518392151,
	0.0131282295669615726370636669025991088987756993,
	0.0121516046710883196351813527366672521223706439,
	0.0111621020998384985912132638285620686108639634,
	0.0101607705350084157575876369538253688489848459,
	0.00914867123078338663258460266520792852644880197,
	0.00812687692569875921738242770785593773424476032,
	0.0070964707911538652691441608121433919347720993,
	0.0060585455042359616833167420317290879695781512,
	0.0050142027429275176924701949690308984740741125,
	0.0039645543384446866737334157674196598776912479,
	0.0029107318179349464084106179894007250097471693,
	0.0018539607889469217323359253508939105882082884,
	7.967920655520124294381434969435687599310869E-4
};

