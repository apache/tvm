/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file random/sgx_random_engine.h
 * \brief SGX trusted random engine
 */
#include <dmlc/logging.h>
#include <sgx_trts.h>
#include <algorithm>
#include <cmath>
#include "../../runtime/sgx/common.h"

namespace tvm {
namespace contrib {

/*!
 * \brief An interface for generating [tensors of] random numbers.
 */
class RandomEngine {
 public:
   /*!
    * \brief Creates a RandomEngine, suggesting the use of a provided seed.
    */
  explicit RandomEngine(unsigned seed) {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
  }

   /*!
    * \brief Seeds the underlying RNG, if possible.
    */
  inline void Seed(unsigned seed) {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
  }

   /*!
    * \return the seed associated with the underlying RNG.
    */
  inline unsigned GetSeed() const {
    LOG(WARNING) << "SGX RandomEngine does not support seeding.";
    return 0;
  }

   /*!
    * \return a random integer sampled from the RNG.
    */
  inline unsigned GetRandInt() {
    int rand_int;
    TVM_SGX_CHECKED_CALL(
        sgx_read_rand(reinterpret_cast<unsigned char*>(&rand_int), sizeof(int)));
    return rand_int;
  }

   /*!
    * \return a random integer sampled from Unif(low, high).
    */
  inline float GetUniform(float low, float high) {
    float max_int = static_cast<float>(std::numeric_limits<unsigned>::max());
    float unif01 = GetRandInt() / max_int;
    return low + unif01 * (high - low);
  }

   /*!
    * \return a random value sampled from Normal(loc, scale**2).
    */
  inline float GetNormal(float loc, float scale) {
    float sign = GetUniform(-1, 1);
    float sample = GetStandardNormalOneside();
    return loc + (sign > 0 ? scale : -scale) * sample;
  }

   /*!
    * \brief Fills a tensor with values drawn from Unif(low, high)
    */
  void SampleUniform(DLTensor* data, float low, float high) {
    CHECK_GT(high, low) << "high must be bigger than low";
    CHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    std::generate_n(static_cast<float*>(data->data), size, [&] () {
      float max_int = static_cast<float>(std::numeric_limits<unsigned>::max());
      float unif01 = GetRandInt() / max_int;
      return low + unif01 * (high - low);
    });
  }

   /*!
    * \brief Fills a tensor with values drawn from Normal(loc, scale)
    */
  void SampleNormal(DLTensor* data, float loc, float scale) {
    CHECK_GT(scale, 0) << "scale must be positive";
    CHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    std::generate_n(static_cast<float*>(data->data), size, [&] () {
      return GetNormal(loc, scale);
    });
  }

 private:
   /*!
    * \return a random value sampled from Normal(0, 1) such that the
    * sampled value is greater than tail
    */
  inline float GetStandardNormalTail(float tail) {
    while (true) {
      float u1 = GetUniform(0, 1);
      float u2 = GetUniform(0, 1);
      float x = - log(u1) / tail;
      float y = - log(u2);
      if (2 * y < x * x) {
        return x + tail;
      }
    }
  }

   /*!
    * \return a random positive value sampled from Normal(0, 1).
    */
  inline float GetStandardNormalOneside() {
    while (true) {
      unsigned i = GetRandInt() & 255;
      float x = GetUniform(0, ZIG_NORM_X[i]);
      if (x < ZIG_NORM_X[i+1]) {
        return x;
      }
      if (i == 0) {
        return GetStandardNormalTail(ZIG_NORM_X[1]);
      }
      float y = GetUniform(ZIG_NORM_F[i], ZIG_NORM_F[i+1]);
      if (y < exp(-0.5 * x * x)) {
        return x;
      }
    }
  }

   /*!
    * Tables for normal distribution which is sampled using the ziggurat algorithm.
    */
  static constexpr float ZIG_NORM_X[257] =
    {3.910757959537090045, 3.654152885361008796, 3.449278298560964462, 3.320244733839166074,
     3.224575052047029100, 3.147889289517149969, 3.083526132001233044, 3.027837791768635434,
     2.978603279880844834, 2.934366867207854224, 2.894121053612348060, 2.857138730872132548,
     2.822877396825325125, 2.790921174000785765, 2.760944005278822555, 2.732685359042827056,
     2.705933656121858100, 2.680514643284522158, 2.656283037575502437, 2.633116393630324570,
     2.610910518487548515, 2.589575986706995181, 2.569035452680536569, 2.549221550323460761,
     2.530075232158516929, 2.511544441625342294, 2.493583041269680667, 2.476149939669143318,
     2.459208374333311298, 2.442725318198956774, 2.426670984935725972, 2.411018413899685520,
     2.395743119780480601, 2.380822795170626005, 2.366237056715818632, 2.351967227377659952,
     2.337996148795031370, 2.324308018869623016, 2.310888250599850036, 2.297723348901329565,
     2.284800802722946056, 2.272108990226823888, 2.259637095172217780, 2.247375032945807760,
     2.235313384928327984, 2.223443340090905718, 2.211756642882544366, 2.200245546609647995,
     2.188902771624720689, 2.177721467738641614, 2.166695180352645966, 2.155817819875063268,
     2.145083634046203613, 2.134487182844320152, 2.124023315687815661, 2.113687150684933957,
     2.103474055713146829, 2.093379631137050279, 2.083399693996551783, 2.073530263516978778,
     2.063767547809956415, 2.054107931648864849, 2.044547965215732788, 2.035084353727808715,
     2.025713947862032960, 2.016433734904371722, 2.007240830558684852, 1.998132471356564244,
     1.989106007615571325, 1.980158896898598364, 1.971288697931769640, 1.962493064942461896,
     1.953769742382734043, 1.945116560006753925, 1.936531428273758904, 1.928012334050718257,
     1.919557336591228847, 1.911164563769282232, 1.902832208548446369, 1.894558525668710081,
     1.886341828534776388, 1.878180486290977669, 1.870072921069236838, 1.862017605397632281,
     1.854013059758148119, 1.846057850283119750, 1.838150586580728607, 1.830289919680666566,
     1.822474540091783224, 1.814703175964167636, 1.806974591348693426, 1.799287584547580199,
     1.791640986550010028, 1.784033659547276329, 1.776464495522344977, 1.768932414909077933,
     1.761436365316706665, 1.753975320315455111, 1.746548278279492994, 1.739154261283669012,
     1.731792314050707216, 1.724461502945775715, 1.717160915015540690, 1.709889657069006086,
     1.702646854797613907, 1.695431651932238548, 1.688243209434858727, 1.681080704722823338,
     1.673943330923760353, 1.666830296159286684, 1.659740822855789499, 1.652674147080648526,
     1.645629517902360339, 1.638606196773111146, 1.631603456932422036, 1.624620582830568427,
     1.617656869570534228, 1.610711622367333673, 1.603784156023583041, 1.596873794420261339,
     1.589979870021648534, 1.583101723393471438, 1.576238702733332886, 1.569390163412534456,
     1.562555467528439657, 1.555733983466554893, 1.548925085471535512, 1.542128153226347553,
     1.535342571438843118, 1.528567729435024614, 1.521803020758293101, 1.515047842773992404,
     1.508301596278571965, 1.501563685112706548, 1.494833515777718391, 1.488110497054654369,
     1.481394039625375747, 1.474683555695025516, 1.467978458615230908, 1.461278162507407830,
     1.454582081885523293, 1.447889631277669675, 1.441200224845798017, 1.434513276002946425,
     1.427828197027290358, 1.421144398672323117, 1.414461289772464658, 1.407778276843371534,
     1.401094763676202559, 1.394410150925071257, 1.387723835686884621, 1.381035211072741964,
     1.374343665770030531, 1.367648583594317957, 1.360949343030101844, 1.354245316759430606,
     1.347535871177359290, 1.340820365893152122, 1.334098153216083604, 1.327368577624624679,
     1.320630975217730096, 1.313884673146868964, 1.307128989027353860, 1.300363230327433728,
     1.293586693733517645, 1.286798664489786415, 1.279998415710333237, 1.273185207661843732,
     1.266358287014688333, 1.259516886060144225, 1.252660221891297887, 1.245787495544997903,
     1.238897891102027415, 1.231990574742445110, 1.225064693752808020, 1.218119375481726552,
     1.211153726239911244, 1.204166830140560140, 1.197157747875585931, 1.190125515422801650,
     1.183069142678760732, 1.175987612011489825, 1.168879876726833800, 1.161744859441574240,
     1.154581450355851802, 1.147388505416733873, 1.140164844363995789, 1.132909248648336975,
     1.125620459211294389, 1.118297174115062909, 1.110938046009249502, 1.103541679420268151,
     1.096106627847603487, 1.088631390649514197, 1.081114409698889389, 1.073554065787871714,
     1.065948674757506653, 1.058296483326006454, 1.050595664586207123, 1.042844313139370538,
     1.035040439828605274, 1.027181966030751292, 1.019266717460529215, 1.011292417434978441,
     1.003256679539591412, 0.995156999629943084, 0.986990747093846266, 0.978755155288937750,
     0.970447311058864615, 0.962064143217605250, 0.953602409875572654, 0.945058684462571130,
     0.936429340280896860, 0.927710533396234771, 0.918898183643734989, 0.909987953490768997,
     0.900975224455174528, 0.891855070726792376, 0.882622229578910122, 0.873271068082494550,
     0.863795545546826915, 0.854189171001560554, 0.844444954902423661, 0.834555354079518752,
     0.824512208745288633, 0.814306670128064347, 0.803929116982664893, 0.793369058833152785,
     0.782615023299588763, 0.771654424216739354, 0.760473406422083165, 0.749056662009581653,
     0.737387211425838629, 0.725446140901303549, 0.713212285182022732, 0.700661841097584448,
     0.687767892786257717, 0.674499822827436479, 0.660822574234205984, 0.646695714884388928,
     0.632072236375024632, 0.616896989996235545, 0.601104617743940417, 0.584616766093722262,
     0.567338257040473026, 0.549151702313026790, 0.529909720646495108, 0.509423329585933393,
     0.487443966121754335, 0.463634336771763245, 0.437518402186662658, 0.408389134588000746,
     0.375121332850465727, 0.335737519180459465, 0.286174591747260509, 0.215241895913273806,
     0.000000000000000000};
  static constexpr float ZIG_NORM_F[257] =
    {0.000477467764586655, 0.001260285930498598, 0.002609072746106363, 0.004037972593371872,
     0.005522403299264754, 0.007050875471392110, 0.008616582769422917, 0.010214971439731100,
     0.011842757857943104, 0.013497450601780807, 0.015177088307982072, 0.016880083152595839,
     0.018605121275783350, 0.020351096230109354, 0.022117062707379922, 0.023902203305873237,
     0.025705804008632656, 0.027527235669693315, 0.029365939758230111, 0.031221417192023690,
     0.033093219458688698, 0.034980941461833073, 0.036884215688691151, 0.038802707404656918,
     0.040736110656078753, 0.042684144916619378, 0.044646552251446536, 0.046623094902089664,
     0.048613553216035145, 0.050617723861121788, 0.052635418276973649, 0.054666461325077916,
     0.056710690106399467, 0.058767952921137984, 0.060838108349751806, 0.062921024437977854,
     0.065016577971470438, 0.067124653828023989, 0.069245144397250269, 0.071377949059141965,
     0.073522973714240991, 0.075680130359194964, 0.077849336702372207, 0.080030515814947509,
     0.082223595813495684, 0.084428509570654661, 0.086645194450867782, 0.088873592068594229,
     0.091113648066700734, 0.093365311913026619, 0.095628536713353335, 0.097903279039215627,
     0.100189498769172020, 0.102487158942306270, 0.104796225622867056, 0.107116667775072880,
     0.109448457147210021, 0.111791568164245583, 0.114145977828255210, 0.116511665626037014,
     0.118888613443345698, 0.121276805485235437, 0.123676228202051403, 0.126086870220650349,
     0.128508722280473636, 0.130941777174128166, 0.133386029692162844, 0.135841476571757352,
     0.138308116449064322, 0.140785949814968309, 0.143274978974047118, 0.145775208006537926,
     0.148286642733128721, 0.150809290682410169, 0.153343161060837674, 0.155888264725064563,
     0.158444614156520225, 0.161012223438117663, 0.163591108232982951, 0.166181285765110071,
     0.168782774801850333, 0.171395595638155623, 0.174019770082499359, 0.176655321444406654,
     0.179302274523530397, 0.181960655600216487, 0.184630492427504539, 0.187311814224516926,
     0.190004651671193070, 0.192709036904328807, 0.195425003514885592, 0.198152586546538112,
     0.200891822495431333, 0.203642749311121501, 0.206405406398679298, 0.209179834621935651,
     0.211966076307852941, 0.214764175252008499, 0.217574176725178370, 0.220396127481011589,
     0.223230075764789593, 0.226076071323264877, 0.228934165415577484, 0.231804410825248525,
     0.234686861873252689, 0.237581574432173676, 0.240488605941449107, 0.243408015423711988,
     0.246339863502238771, 0.249284212419516704, 0.252241126056943765, 0.255210669955677150,
     0.258192911338648023, 0.261187919133763713, 0.264195763998317568, 0.267216518344631837,
     0.270250256366959984, 0.273297054069675804, 0.276356989296781264, 0.279430141762765316,
     0.282516593084849388, 0.285616426816658109, 0.288729728483353931, 0.291856585618280984,
     0.294997087801162572, 0.298151326697901342, 0.301319396102034120, 0.304501391977896274,
     0.307697412505553769, 0.310907558127563710, 0.314131931597630143, 0.317370638031222396,
     0.320623784958230129, 0.323891482377732021, 0.327173842814958593, 0.330470981380537099,
     0.333783015832108509, 0.337110066638412809, 0.340452257045945450, 0.343809713148291340,
     0.347182563958251478, 0.350570941482881204, 0.353974980801569250, 0.357394820147290515,
     0.360830600991175754, 0.364282468130549597, 0.367750569780596226, 0.371235057669821344,
     0.374736087139491414, 0.378253817247238111, 0.381788410875031348, 0.385340034841733958,
     0.388908860020464597, 0.392495061461010764, 0.396098818517547080, 0.399720314981931668,
     0.403359739222868885, 0.407017284331247953, 0.410693148271983222, 0.414387534042706784,
     0.418100649839684591, 0.421832709231353298, 0.425583931339900579, 0.429354541031341519,
     0.433144769114574058, 0.436954852549929273, 0.440785034667769915, 0.444635565397727750,
     0.448506701509214067, 0.452398706863882505, 0.456311852680773566, 0.460246417814923481,
     0.464202689050278838, 0.468180961407822172, 0.472181538469883255, 0.476204732721683788,
     0.480250865911249714, 0.484320269428911598, 0.488413284707712059, 0.492530263646148658,
     0.496671569054796314, 0.500837575128482149, 0.505028667945828791, 0.509245245998136142,
     0.513487720749743026, 0.517756517232200619, 0.522052074674794864, 0.526374847174186700,
     0.530725304406193921, 0.535103932383019565, 0.539511234259544614, 0.543947731192649941,
     0.548413963257921133, 0.552910490428519918, 0.557437893621486324, 0.561996775817277916,
     0.566587763258951771, 0.571211506738074970, 0.575868682975210544, 0.580559996103683473,
     0.585286179266300333, 0.590047996335791969, 0.594846243770991268, 0.599681752622167719,
     0.604555390700549533, 0.609468064928895381, 0.614420723892076803, 0.619414360609039205,
     0.624450015550274240, 0.629528779928128279, 0.634651799290960050, 0.639820277456438991,
     0.645035480824251883, 0.650298743114294586, 0.655611470583224665, 0.660975147780241357,
     0.666391343912380640, 0.671861719900766374, 0.677388036222513090, 0.682972161648791376,
     0.688616083008527058, 0.694321916130032579, 0.700091918140490099, 0.705928501336797409,
     0.711834248882358467, 0.717811932634901395, 0.723864533472881599, 0.729995264565802437,
     0.736207598131266683, 0.742505296344636245, 0.748892447223726720, 0.755373506511754500,
     0.761953346841546475, 0.768637315803334831, 0.775431304986138326, 0.782341832659861902,
     0.789376143571198563, 0.796542330428254619, 0.803849483176389490, 0.811307874318219935,
     0.818929191609414797, 0.826726833952094231, 0.834716292992930375, 0.842915653118441077,
     0.851346258465123684, 0.860033621203008636, 0.869008688043793165, 0.878309655816146839,
     0.887984660763399880, 0.898095921906304051, 0.908726440060562912, 0.919991505048360247,
     0.932060075968990209, 0.945198953453078028, 0.959879091812415930, 0.977101701282731328,
     1.000000000000000000};
};

constexpr float RandomEngine::ZIG_NORM_X[];
constexpr float RandomEngine::ZIG_NORM_F[];

}  // namespace contrib
}  // namespace tvm
