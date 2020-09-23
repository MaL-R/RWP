function header=extract_header(entete)
% function header=read_header(fid)
% must be positionned at begining of header
% read complete header information
% ... next read will access moments, spectra or time series ...
% file was previously opened using fid=fopen(filenamestring);
    header.NPAR=entete(1);
    NWDREC1=entete(2);
    header.NHTS=entete(3);
    header.NRX=entete(4);
    header.NPTST_VENT=entete(5);
    header.NSPEC_VENT=entete(6);
    header.NCI_VENT=entete(7);
    header.IPP=entete(8);
    header.PW=entete(9);
    % retard du systême 1: delay to first gate in clock cycles
    header.DLY1=entete(10);
    % espacement du systême 1; range gate spacing in clock cycles
    header.SPAC1=entete(11);
    % nombre de mesures du systême 1; number of range gates
    header.NSAM1=entete(12);
    % retard du systême 2 not used
    header.DLY2=entete(13);
    % espacement du systême 2 not used
    header.SPAC2=entete(14);
    % nombre de mesures du systême 2 not used
    header.NSAM2=entete(15);

    % année de la mesure                 
    header.RECYR=entete(16);
    % jour de la mesure
    header.RECDAY=entete(17);
    % heure de la mesure
    header.RECHR=entete(18);
    % minute de la mesure
    header.RECMIN=entete(19);
    % seconde de la mesure
    header.RECSEC=entete(20);

    header.RECDAYMON=entete(21);
    header.RECMON=entete(22);

    % option de filtre continu ; number of points of the DC filter
    header.DCOPT_VENT=entete(23);
    % fenêtre d'apodisation; use of the weighting window
    header.WDOPT = entete(24);

    % azimut du faisceau                                                                
    header.AZ = entete(25)/10;
    % radar frequency 
    FREQ1 = entete(26);
    % numéro du faisceau	/* BEAM CODE */
    BMCODE = entete(27);
    % altitude du site
    ALT = entete(28);
    % élévation du faisceau
    header.EL = entete(29)/10;

    % délai du systême
    header.SYS_DELAY1=entete(30);
    header.SYS_DELAY2=entete(31);

% if 1 then defaut;fault
% bit 0 (LSB): defaut +5V
% bit 1: defaut +15V
% bit 2: defaut -5V
% bit 3: defaut -15V
% bit 4: defaut thermique
% bit 5: préchauffage
% bit 6: defaut TOS
    RECERR=entete(32);

    % horloge in ns
    header.CLOCK_ns=entete(33);
    % numéro du site
    header.SITE=entete(34);
    % longitude et latitude en degré, minute, seconde
    header.LONGITUDE=entete(35);
    header.LONGIT_MIN=entete(36);
    header.LONGIT_SEC=entete(37);
    header.LATITUDE=entete(38);
    header.LAT_MIN=entete(39);
    header.LAT_SEC=entete(40);

    header.CODE_NBR=entete(41);
    header.CODE_NBR_IPP=entete(42);
    header.CODE_NBR_MOM=entete(43);
    header.CODE_NBR_LIGNE=entete(44);
    header.DECODE_TRONC=entete(45);

    % nombre de pics calculés; the number of peaks 
    header.N_PIC_VENT=entete(46);

    % VHF/1 ou UHF/0
    header.MOD_ST=entete(47);

    % puissance crête de l'émetteur en Watts (peak power of transmitter)
    header.PUIS_TX=entete(48);

    % version de format des données
    header.VERS_DATA=entete(49);

    % présence du tableau de qualité et compression des données
    % format of data: floats compressed or not
    %#define TAB_QUALITE		0x0001	// présence du tableau qualité
    %#define COMP_FLOAT         0x0002	// compression des tableaux de moments et de spectres
	%#define PC_FLOAT           0x0004	// Flottant DSP ont été transformé au format PC; floatting point numbers are IEEE instead of DSP32C
	%#define SERIE_TRIE         0x0008	// séries temporelles rangées porte par porte au lieu de spectre par spectre; time series data hase been sorted
	%#define TEMPORELLE_CODE	0x00010	// la série temporelle n'a pas été décodé; time series has not been decoded
	%#define TEMPORELLE_FPGA	0x00020	// la série temporelle  acquise par nouveau bloque numérique= avant integration coherente SOFT
	%									// (acquisition via DSP: intégration cohérente déjà faite -> deux série temporelles VENT et SON)
    header.FORMAT=entete(50);

    % version du programme DSP (version of DSP programme)
    header.VERS_DSP=entete(51);

    % details of signal processing
    % algorithmes utilisés par la DSP (cf DSPFILE.S/lAlgo)
    %#define AVEC_MERRITT	0x0001
    %#define CORR_QUADRA		0x0002
    %#define SKEW			0x0004
    %#define CORR_OFFSET		0x0008
    header.ALGO_DSP=entete(52);

    % recouvrement spectral (en points)
    % number of points in the spectral overlap
    header.RECOUVREMENT_VENT=entete(53);

    % durée réelle de la pulse (peut être différent de la durée transmise à l'émetteur)
    % en nanosecondes
    % real pulse length at ouptut of transmitter in nanoseconds
    header.PULSE_TX_ns=entete(54);

    % détection de la pluie sur la verticale du cycle précédent
    % rain detection by PCA on vertical beam of preceding cycle
    header.PLUIE=entete(55);

    % pour calibration; info antenne en 1/10 dB
    % to calculate dBZ information
    header.GAIN_ANT=entete(56);
    header.TX_LOSS_ANT=entete(57)/10;
    header.RX_LOSS_ANT=entete(58)/10;

    %	58 à 60     
    header.RECMSSEC=entete(59);

    % concatenation limits
    header.CONCAT_LIMIT_VENT=entete(60);
    header.CONCAT_LIMIT_SON=entete(61);


    % Partie haute du nb de mot entête + mesure ( passage en 32 bits )
    % upper 16 bit of record size
    NWDREC_HIGH=entete(62);

    % NCI pour la partie SON ( Si = 0, ceci n'est pas une donnée RASS )
    % NCI for sound in RASS systems
    NCI_SON=entete(63);

    % NCI pour la carte intégrateur ( Si = 0, cf ancienne mesure avec 1 seul NCI -> Mettre NCI_HARD = NCI )
    % NCI send to integrator board 
    % in old files if =0 then set NCI_HARD = NCI
    NCI_HARD=entete(64);

    % nombre de points par spectre
    % number of spectral points for sound spectra in RASS systems
    header.NPTST_SON=entete(65);

    % nombre de moyennes = nombre d'intégrations incohérentes du signal
    % number of incoherent averages of the spound spectra for RASS systems
    header.NSPEC_SON=entete(66);

    % option de filtre continu
    % DC Filter for sound
    header.DCOPT_SON=entete(67);

    % nombre de pics calculés
    % number of peaks to be calculated for sound
    header.N_PIC_SON=entete(68);

    % recouvrement spectral (en points)
    % sound spectral overlap
    header.RECOUVREMENT_SON=entete(69);

    FREQ_HIGH=entete(70);


% facteur de bruit du récepteur en 0.1 dB
% receiver noise figure in 0.1 dB
    header.RX_NF_dB=entete(71)/10;
% elements des convertisseurs Ana -> Num
% necessaire pour "calibrer" le radar
    header.ADC_BITS=entete(72);
    header.ADC_SCALE=entete(73);
    header.ADC_IMPEDANCE=entete(74);


    % a partir de la version 5:
    % starting version 5
    header.TX_CURRENT_mA=entete(75);
    header.ATTENUATION_dB=entete(76);

    %#define CHAMP_ENTETE_INVALIDE 10000	// valeur invalide
    % ground station measurements
    header.ROSE_AURIA=entete(77);
    header.VENT_AURIA=entete(78);
    header.PLUIE_AURIA=entete(79);
    header.TEMP_AURIA=entete(80);
    header.HUMID_AURIA=entete(81);
    header.PRESSION_AURIA=entete(82);

    % nouvelle info data version posterieur à 3 -> codage BUFR
    header.WMO_BLOCK=entete(83);
    header.OWNER_COUNTRY=entete(84);
    header.OWNER_AGENCY=entete(85);
    header.INSTRUMENT=entete(86);
    header.ANTENNA_TYPE=entete(87);
    header.BEAMWIDTH=entete(88)/10;
    header.STATION_TYPE=entete(89);
    % non définis jusqu'à MAXHEAD
    % difference between local time and UTC/ZU/GMT in minutes
    % starting version 6
    header.FUSEAU_HORAIRE_MIN=entete(90);


    header.NWDREC = 65536 * NWDREC_HIGH + NWDREC1 ;
    header.FREQ = 65536 * FREQ_HIGH + FREQ1;
    header.NCI_WIND = header.NCI_VENT * NCI_HARD;
    header.NCI_SON = NCI_SON * NCI_HARD;
    header.PW = header.PW * header.CLOCK_ns/1000;
    header.IPP = header.IPP * header.CLOCK_ns/1000;
    header.SPAC1 = header.SPAC1 * header.CLOCK_ns/1000;
        
    % do house keeping
    if bitand(header.FORMAT,2)==2
        header.CompressedFloat = 1;
        nCoef=2;
    else
        header.CompressedFloat = 0;
        nCoef = 1;
    end        
    if bitand(header.FORMAT,4)==4
        header.PC_Float = 1;
    else
        header.PC_Float = 0;
    end        
    if bitand(header.FORMAT,8)==8
        header.Sorted_Time_Series = 1;
    else
        header.Sorted_Time_Series = 0;
    end        
    if bitand(header.FORMAT,16)==16
        header.Coded_Time_Series = 1;
        nIPP = header.CODE_NBR_IPP;
    else
        header.Coded_Time_Series = 0;
        nIPP = 1;
    end        
    nb_pointParPorte = (header.NPTST_VENT - header.RECOUVREMENT_VENT)*header.NSPEC_VENT + (header.NPTST_SON - header.RECOUVREMENT_SON)*header.NSPEC_SON;
    taille_serietemp = header.NPAR * 2 + header.NHTS *(  nb_pointParPorte * nIPP * 2 * (4/nCoef));
    if taille_serietemp == header.NWDREC
        % c'est une série temporelle pure
        header.bSerieTemporelle = 1;
        header.bMoments = 0;
        header.bSpectres=0;
    else
        % il y a au moins les moments
        header.bMoments = 1;
        % est ce qu'il y a les points spectraux ?
        taille_moment = header.NPAR * 2 + header.NHTS *((header.N_PIC_VENT*4+1)+(header.N_PIC_SON*4+1))* 4/nCoef;
        if header.NWDREC > taille_moment
            % il y a aussi les spectres
            header.bSpectres=1;
            taille_spectre = taille_moment +  header.NHTS * (header.NPTST_VENT + header.NPTST_SON)* 4/nCoef;
            if header.NWDREC > taille_spectre
                header.bSerieTemporelle = 1;
            else
                header.bSerieTemporelle = 0;
            end
            
        else
            % il n'y a que les moments
            header.bSpectres=0;
            header.bSerieTemporelle = 0;
        end
        
    end
    
    