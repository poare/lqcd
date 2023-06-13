###########################################################################
# This script will compute the coefficients to the rational expansion     #
# r(K) = \alpha_0 + \sum_{i = 1}^P \frac{\alpha_i}{K + \beta_i}           #
# to the polynomial r(K)\approx K^{-\gamma}, with \gamma = 1/4 or 1/8.    # 
# This uses the AlgRemez library to compute the necessary approximation   #
# coefficients, https://github.com/maddyscientist/AlgRemez.               #
# Main functions to use are 5, 6, 8, 9, and 15.                           #
###########################################################################

import numpy as np

def rhmc_m4_19():
    """
    Approximates r(K) = K^{-1/4} with 19 partial fractions. Approximation is valid for 
    lambda_min = 1e-8 and lambda_max = 50000. This is the output of running 
    ```./test 1 4 19 19 1e-8 50000 128``` in AlgRemez.

    Error for n_root = 1 is 1.680918e-05. Note the size of the coefficients are P + 1 = 20, 
    since we're including alpha_0 in the coefficients.
    """
    alpha, beta = np.zeros((20), dtype = np.float64), np.zeros((20), dtype = np.float64)
    alpha[0] = 3.5781609297300059e-02
    alpha[1] = 3.0717855433550755e-07
    alpha[2] = 1.0029877194364809e-06
    alpha[3] = 3.3516396678877979e-06
    alpha[4] = 1.1410873683384280e-05
    alpha[5] = 3.9010444318468622e-05
    alpha[6] = 1.3347525170664460e-04
    alpha[7] = 4.5676237254114321e-04
    alpha[8] = 1.5631239828206335e-03
    alpha[9] = 5.3493273393033745e-03
    alpha[10] = 1.8306508430273444e-02
    alpha[11] = 6.2648751795464655e-02
    alpha[12] = 2.1439848304452658e-01
    alpha[13] = 7.3374167738774820e-01
    alpha[14] = 2.5114673617263459e+00
    alpha[15] = 8.6027269429096123e+00
    alpha[16] = 2.9581317991456945e+01
    alpha[17] = 1.0376274631091705e+02
    alpha[18] = 4.0372747959857907e+02
    alpha[19] = 2.7865429990752900e+03

    beta[1] = 3.0177513487338543e-09
    beta[2] = 3.1640198707444036e-08
    beta[3] = 1.8284145486115925e-07
    beta[4] = 9.6330653865743114e-07
    beta[5] = 4.9884059156280029e-06
    beta[6] = 2.5746412021593944e-05
    beta[7] = 1.3279825136539079e-04
    beta[8] = 6.8487901396126834e-04
    beta[9] = 3.5320339813338882e-03
    beta[10] = 1.8215198273083606e-02
    beta[11] = 9.3938322348130904e-02
    beta[12] = 4.8445447391345675e-01
    beta[13] = 2.4984467253127556e+00
    beta[14] = 1.2886148723664553e+01
    beta[15] = 6.6490790460121033e+01
    beta[16] = 3.4384001412588748e+02
    beta[17] = 1.7984908711932201e+03
    beta[18] = 9.9925026080852913e+03
    beta[19] = 7.9423937552225543e+04

    return alpha, beta

def rhmc_8_19():
    """
    Approximates r(K) = K^{+1/8} with 19 partial fractions. Approximation is valid for 
    lambda_min = 1e-8 and lambda_max = 50000. This is the output of running 
    ```./test 1 8 19 19 1e-8 50000 128``` in AlgRemez.

    Error for n_root = 1 is 9.075152e-06.  
    """
    alpha, beta = np.zeros((20), dtype = np.float64), np.zeros((20), dtype = np.float64)

    alpha[0] = 5.2602639578072914e+00
    alpha[1] = -1.6579767875762553e-10
    beta[1] = 5.3667966808839825e-09
    alpha[2] = -1.1933378657034187e-09
    beta[2] = 4.4848665745573186e-08
    alpha[3] = -7.6491763360273912e-09
    beta[3] = 2.5104164739782321e-07
    alpha[4] = -4.8506632334522925e-08
    beta[4] = 1.3145726573450043e-06
    alpha[5] = -3.0708961697783274e-07
    beta[5] = 6.7976431788971964e-06
    alpha[6] = -1.9435505815603433e-06
    beta[6] = 3.5065304575557148e-05
    alpha[7] = -1.2299882595149823e-05
    beta[7] = 1.8079754545412754e-04
    alpha[8] = -7.7839696558059101e-05
    beta[8] = 9.3211140042029372e-04
    alpha[9] = -4.9260676303235261e-04
    beta[9] = 4.8054651995520668e-03
    alpha[10] = -3.1174504955822726e-03
    beta[10] = 2.4774313629046037e-02
    alpha[11] = -1.9728745915332670e-02
    beta[11] = 1.2772264147059928e-01
    alpha[12] = -1.2485423427640822e-01
    beta[12] = 6.5846989849931281e-01
    alpha[13] = -7.9018197667794854e-01
    beta[13] = 3.3947936807672341e+00
    alpha[14] = -5.0021225796272590e+00
    beta[14] = 1.7504087935332475e+01
    alpha[15] = -3.1704032339127874e+01
    beta[15] = 9.0305963859099236e+01
    alpha[16] = -2.0221992733851556e+02
    beta[16] = 4.6729184698558464e+02
    alpha[17] = -1.3327569152018057e+03
    beta[17] = 2.4557178121577372e+03
    alpha[18] = -1.0428797473207418e+04
    beta[18] = 1.4015074533288203e+04
    alpha[19] = -2.2083388890882154e+05
    beta[19] = 1.3399028714880449e+05

    return alpha, beta

def rhmc_m4_15():
    """
    Approximates r(K) = K^{-1/4} with 15 partial fractions. Approximation is valid for 
    lambda_min = 1e-7 and lambda_max = 1000. This is the output of running 
    ```./test 1 4 15 15 1e-7 1000 128``` in AlgRemez.

    Error for n_root = 1 is 1.975834e-05.
    """
    alpha, beta = np.zeros((16), dtype = np.float64), np.zeros((16), dtype = np.float64)

    alpha[0] = 9.5797060554725838e-02
    alpha[1] = 1.7701746700099842e-06
    beta[1] = 3.1085594175442315e-08
    alpha[2] = 5.8705983656937455e-06
    beta[2] = 3.2994455960441383e-07
    alpha[3] = 1.9961158693570120e-05
    beta[3] = 1.9424842756552213e-06
    alpha[4] = 6.9125367600088173e-05
    beta[4] = 1.0453359626231250e-05
    alpha[5] = 2.4032965323696816e-04
    beta[5] = 5.5337819905761986e-05
    alpha[6] = 8.3620125835371663e-04
    beta[6] = 2.9204178440857227e-04
    alpha[7] = 2.9099006745502945e-03
    beta[7] = 1.5403300046437174e-03
    alpha[8] = 1.0126504714418652e-02
    beta[8] = 8.1233558140562465e-03
    alpha[9] = 3.5241454044660878e-02
    beta[9] = 4.2840454273820550e-02
    alpha[10] = 1.2266034741624667e-01
    beta[10] = 2.2594500626442715e-01
    alpha[11] = 4.2721681852328125e-01
    beta[11] = 1.1921171782283737e+00
    alpha[12] = 1.4932820692676758e+00
    beta[12] = 6.3026182343759860e+00
    alpha[13] = 5.3188766358452595e+00
    beta[13] = 3.3683411978650057e+01
    alpha[14] = 2.0944763089672641e+01
    beta[14] = 1.9083658214156412e+02
    alpha[15] = 1.4525770103354523e+02
    beta[15] = 1.5386784635765257e+03

    return alpha, beta

def rhmc_8_15():
    """
    Approximates r(K) = K^{+1/8} with 15 partial fractions. Approximation is valid for 
    lambda_min = 1e-7 and lambda_max = 1000. This is the output of running 
    ```./test 1 8 15 15 1e-7 1000 128``` in AlgRemez.

    Error for n_root = 1 is 1.066076e-05.
    """
    alpha, beta = np.zeros((16), dtype = np.float64), np.zeros((16), dtype = np.float64)

    alpha[0] = 3.2148873149863206e+00
    alpha[1] = -2.2977600408751347e-09
    beta[1] = 5.5367335615411457e-08
    alpha[2] = -1.6898103706901080e-08
    beta[2] = 4.6910257304582898e-07
    alpha[3] = -1.1099658368596435e-07
    beta[3] = 2.6768223190551614e-06
    alpha[4] = -7.2162146587729939e-07
    beta[4] = 1.4319657256375662e-05
    alpha[5] = -4.6841070484595924e-06
    beta[5] = 7.5694473187855338e-05
    alpha[6] = -3.0396303865820386e-05
    beta[6] = 3.9922490005559548e-04
    alpha[7] = -1.9723870959636086e-04
    beta[7] = 2.1046795395127538e-03
    alpha[8] = -1.2798599250624021e-03
    beta[8] = 1.1094832053548640e-02
    alpha[9] = -8.3051856063983548e-03
    beta[9] = 5.8486687698920667e-02
    alpha[10] = -5.3904877281192094e-02
    beta[10] = 3.0834388405073770e-01
    alpha[11] = -3.5026088217184553e-01
    beta[11] = 1.6264534005778293e+00
    alpha[12] = -2.2893521967679966e+00
    beta[12] = 8.6030459456576764e+00
    alpha[13] = -1.5436668340425719e+01
    beta[13] = 4.6179583183155444e+01
    alpha[14] = -1.2297861076048798e+02
    beta[14] = 2.6854965277696181e+02
    alpha[15] = -2.6252652966414048e+03
    beta[15] = 2.6004158696112045e+03

    return alpha, beta

def rhmc_m4_9():
    """
    Approximates r(K) = K^{-1/4} with 9 partial fractions. Approximation is valid for 
    lambda_min = 1e-4 and lambda_max = 45. This is the output of running 
    ```./test 1 4 9 9 1e-4 45 128``` in AlgRemez.

    Error for n_root = 1 is 1.928152e-05.
    """
    alpha, beta = np.zeros((10), dtype = np.float64), np.zeros((10), dtype = np.float64)

    alpha[0] = 2.0777859903387064e-01
    alpha[1] = 3.1361646774095720e-04
    beta[1] = 3.0945740855689258e-05
    alpha[2] = 1.0376047912755866e-03
    beta[2] = 3.2784330944557943e-04
    alpha[3] = 3.5188116165401893e-03
    beta[3] = 1.9246508827843313e-03
    alpha[4] = 1.2155937785551100e-02
    beta[4] = 1.0324786841812992e-02
    alpha[5] = 4.2190013023882304e-02
    beta[5] = 5.4499729309375029e-02
    alpha[6] = 1.4707389241596666e-01
    beta[6] = 2.8737303500233369e-01
    alpha[7] = 5.2261160990536404e-01
    beta[7] = 1.5309893112867550e+00
    alpha[8] = 2.0541440716938171e+00
    beta[8] = 8.6482893683193165e+00
    alpha[9] = 1.4235435645059507e+01
    beta[9] = 6.9576998834492443e+01

    return alpha, beta


def rhmc_8_9():
    """
    Approximates r(K) = K^{+1/8} with 9 partial fractions. Approximation is valid for 
    lambda_min = 1e-4 and lambda_max = 45. This is the output of running 
    ```./test 1 8 9 9 1e-4 45 128``` in AlgRemez.

    Error for n_root = 1 is 1.038366e-05.
    """
    alpha, beta = np.zeros((10), dtype = np.float64), np.zeros((10), dtype = np.float64)

    alpha[0] = 2.1830271620728054e+00
    alpha[1] = -5.4143795366382000e-06
    beta[1] = 5.5083850325016191e-05
    alpha[2] = -3.9676818015731586e-05
    beta[2] = 4.6563686474799176e-04
    alpha[3] = -2.5958109726990279e-04
    beta[3] = 2.6486486150264711e-03
    alpha[4] = -1.6811247632604292e-03
    beta[4] = 1.4120708057479768e-02
    alpha[5] = -1.0882543862047360e-02
    beta[5] = 7.4420558484424110e-02
    alpha[6] = -7.0847423319728930e-02
    beta[6] = 3.9242609363745878e-01
    alpha[7] = -4.7591272677901386e-01
    beta[7] = 2.0993041046459315e+00
    alpha[8] = -3.7801496263213803e+00
    beta[8] = 1.2170425789307121e+01
    alpha[9] = -8.0587284363165352e+01
    beta[9] = 1.1759944538524526e+02

    return alpha, beta

def rhmc_m4_8():
    """
    Approximates r(K) = K^{-1/4} with 8 partial fractions. Approximation is valid for 
    lambda_min = 1e-3 and lambda_max = 50. This is the output of running 
    ```./test 1 4 8 8 1e-3 50 128``` in AlgRemez.

    Error for n_root = 1 is 1.205192e-05.
    """
    alpha, beta = np.zeros((9), dtype = np.float64), np.zeros((9), dtype = np.float64)

    alpha[0] = 1.9847855485120461e-01
    alpha[1] = 1.6447767692749293e-03
    beta[1] = 2.8432460169357867e-04
    alpha[2] = 5.2091545016137450e-03
    beta[2] = 2.9108780584128227e-03
    alpha[3] = 1.6823654204678816e-02
    beta[3] = 1.6221830544213445e-02
    alpha[4] = 5.5444365469241169e-02
    beta[4] = 8.2026799734265451e-02
    alpha[5] = 1.8437888053401108e-01
    beta[5] = 4.0801052401031673e-01
    alpha[6] = 6.2746826716341830e-01
    beta[6] = 2.0476062496901357e+00
    alpha[7] = 2.3852840251249825e+00
    beta[7] = 1.0951562209548896e+01
    alpha[8] = 1.6315143889652543e+01
    beta[8] = 8.4659732253886020e+01

    return alpha, beta

def rhmc_8_8():
    """
    Approximates r(K) = K^{+1/8} with 8 partial fractions. Approximation is valid for 
    lambda_min = 1e-3 and lambda_max = 50. This is the output of running 
    ```./test 1 8 8 8 1e-3 50 128``` in AlgRemez.

    Error for n_root = 1 is 6.486560e-06.
    """
    alpha, beta = np.zeros((9), dtype = np.float64), np.zeros((9), dtype = np.float64)

    alpha[0] = 2.2336270511419518e+00
    alpha[1] = -6.4667060419709420e-05
    beta[1] = 5.0374382496533006e-04
    alpha[2] = -4.4599197309502808e-04
    beta[2] = 4.0970063350602185e-03
    alpha[3] = -2.7234661408815967e-03
    beta[3] = 2.2077737619020866e-02
    alpha[4] = -1.6461958362314753e-02
    beta[4] = 1.1088619746225478e-01
    alpha[5] = -1.0004750758332101e-01
    beta[5] = 5.5108791671543900e-01
    alpha[6] = -6.3021119918448276e-01
    beta[6] = 2.7782277485822364e+00
    alpha[7] = -4.7606598099602264e+00
    beta[7] = 1.5278036140176734e+01
    alpha[8] = -9.9243921490085825e+01
    beta[8] = 1.4244275240229373e+02

    return alpha, beta

def rhmc_m4_6():
    """
    Approximates r(K) = K^{-1/4} with 6 partial fractions. Approximation is valid for 
    lambda_min = 0.02 and lambda_max = 50. This is the output of running 
    ```./test 1 4 6 6 0.02 50 128``` in AlgRemez.

    Error for n_root = 1 is 1.515132e-05.
    """
    alpha, beta = np.zeros((7), dtype = np.float64), np.zeros((7), dtype = np.float64)

    alpha[0] = 2.0035838418319055e-01
    alpha[1] = 1.6089619700034339e-02
    beta[1] = 5.9246187607994976e-03
    alpha[2] = 5.2075939216247140e-02
    beta[2] = 6.1679058754489605e-02
    alpha[3] = 1.7299810700423821e-01
    beta[3] = 3.5332103905499312e-01
    alpha[4] = 5.9783724002907257e-01
    beta[4] = 1.8640825764563929e+00
    alpha[5] = 2.3063654482894740e+00
    beta[5] = 1.0275478485280116e+01
    alpha[6] = 1.5868596323671357e+01
    beta[6] = 8.1016349994659194e+01

    return alpha, beta

def rhmc_8_6():
    """
    Approximates r(K) = K^{+1/8} with 6 partial fractions. Approximation is valid for 
    lambda_min = 0.02 and lambda_max = 50. This is the output of running 
    ```./test 1 8 6 6 0.02 50 128``` in AlgRemez.

    Error for n_root = 1 is 8.140949e-06.
    """
    alpha, beta = np.zeros((7), dtype = np.float64), np.zeros((7), dtype = np.float64)

    alpha[0] = 2.2231956780463324e+00
    alpha[1] = -1.9829856353425169e-03
    beta[1] = 1.0516496796568211e-02
    alpha[2] = -1.4094809881415831e-02
    beta[2] = 8.7160697505863152e-02
    alpha[3] = -8.9639421822227741e-02
    beta[3] = 4.8363725652222034e-01
    alpha[4] = -5.8322796414232880e-01
    beta[4] = 2.5473481002404794e+00
    alpha[5] = -4.5123946696300674e+00
    beta[5] = 1.4405517772037626e+01
    alpha[6] = -9.5059971756102911e+01
    beta[6] = 1.3666238033906589e+02

    return alpha, beta

def rhmc_m4_5():
    """
    Approximates r(K) = K^{-1/4} with 5 partial fractions. Approximation is valid for 
    lambda_min = 0.1 and lambda_max = 50. This is the output of running 
    ```./test 1 4 5 5 0.1 50 128``` in AlgRemez.

    Error for n_root = 1 is 1.548364e-05.
    """
    alpha, beta = np.zeros((6), dtype = np.float64), np.zeros((6), dtype = np.float64)

    alpha[0] = 2.0057814828187764e-01
    alpha[1] = 5.4026803430958975e-02
    beta[1] = 2.9769772029903337e-02
    alpha[2] = 1.7590005470348138e-01
    beta[2] = 3.1110339537434900e-01
    alpha[3] = 5.9772503229277774e-01
    beta[3] = 1.8074132732436818e+00
    alpha[4] = 2.2997285767889570e+00
    beta[4] = 1.0165968473383909e+01
    alpha[5] = 1.5819178342283395e+01
    beta[5] = 8.0573715946619146e+01

    return alpha, beta

def rhmc_8_5():
    """
    Approximates r(K) = K^{+1/8} with 5 partial fractions. Approximation is valid for 
    lambda_min = 0.1 and lambda_max = 50. This is the output of running 
    ```./test 1 8 5 5 0.1 50 128``` in AlgRemez.

    Error for n_root = 1 is 8.308677e-06.
    """
    alpha, beta = np.zeros((6), dtype = np.float64), np.zeros((6), dtype = np.float64)

    alpha[0] = 2.2220375827770185e+00
    alpha[1] = -1.2204111391487278e-02
    beta[1] = 5.2850306687580832e-02
    alpha[2] = -8.7538464213535880e-02
    beta[2] = 4.4009623019926597e-01
    alpha[3] = -5.7743852008615837e-01
    beta[3] = 2.4868497380830941e+00
    alpha[4] = -4.4849784135647406e+00
    beta[4] = 1.4278758280275570e+01
    alpha[5] = -9.4603555440254667e+01
    beta[5] = 1.3600523930215593e+02

    return alpha, beta