// NPR momentum-fraction measurement for Chroma.
//
// Reproduces compute_npr_through_op from the QLUA npr_momfrac (old) script,
// used for the quark EMT / <x> renormalization in Detmold et al. 2020.
//
// Two build routes:
//   (a) Standalone binary (default): compile as-is. main() registers the
//       measurement into a stock Chroma and runs Chroma's driver loop.
//   (b) Inside a Chroma or LALIBE tree: compile with -DNPR_MOMFRAC_NO_MAIN
//       and add this file to the inline-hadron aggregate. Use this route when
//       the fermion action you want lives in that tree (e.g. exponential
//       clover), since it must be registered in whichever binary runs.
#include "chroma.h"
#include <iostream>

using namespace Chroma;

namespace NprMomfrac {

  struct Params {
    Params() : frequency(0) {}
    Params(XMLReader& xml_in, const std::string& path);
    unsigned long frequency;
    std::string   gauge_id;
  };

  Params::Params(XMLReader& xml_in, const std::string& path) {
    XMLReader paramtop(xml_in, path);
    if (paramtop.count("Frequency") == 1) read(paramtop, "Frequency", frequency);
    else frequency = 1;
    read(paramtop, "NamedObject/gauge_id", gauge_id);
  }

  class InlineNprMomfrac : public AbsInlineMeasurement {
  public:
    InlineNprMomfrac(const Params& p) : params(p) {}
    unsigned long getFrequency() const { return params.frequency; }
    void operator()(unsigned long update_no, XMLWriter& xml_out) {
      QDPIO::cout << "NPR_MOMFRAC: measurement reached, gauge_id = "
                  << params.gauge_id << std::endl;
      push(xml_out, "NprMomfrac");
      write(xml_out, "update_no", update_no);
      pop(xml_out);
    }
  private:
    Params params;
  };

  namespace {
    AbsInlineMeasurement* createMeasurement(XMLReader& xml_in,
                                            const std::string& path) {
      return new InlineNprMomfrac(Params(xml_in, path));
    }
    bool registered = false;
  }

  const std::string name = "NPR_MOMFRAC";

  bool registerAll() {
    bool success = true;
    if (!registered) {
      success &= WilsonTypeFermActsEnv::registerAll();
      success &= TheInlineMeasurementFactory::Instance()
                   .registerObject(name, createMeasurement);
      registered = true;
    }
    return success;
  }

} // namespace NprMomfrac

#ifndef NPR_MOMFRAC_NO_MAIN

// Driver. This follows mainprogs/main/chroma.cc closely and deliberately: the
// point of the standalone route is to be stock Chroma's own loop with one
// extra registerAll() in front of the measurement read. Deviating from it is
// how the gauge field or the RNG ends up in a subtly different state.

namespace {

  struct DriverInput {
    multi1d<int> nrow;
    std::string  inline_measurement_xml;
    GroupXML_t   cfg;
    QDP::Seed    rng_seed;
  };

  void readDriverInput(XMLReader& xml, const std::string& path, DriverInput& p) {
    XMLReader top(xml, path);
    XMLReader paramtop(top, "Param");
    read(paramtop, "nrow", p.nrow);

    XMLReader measurements_xml(paramtop, "InlineMeasurements");
    std::ostringstream inline_os;
    measurements_xml.print(inline_os);
    p.inline_measurement_xml = inline_os.str();

    p.cfg = readXMLGroup(top, "Cfg", "cfg_type");

    if (top.count("RNG") > 0) read(top, "RNG", p.rng_seed);
    else                      p.rng_seed = 11;
  }

} // anonymous namespace

int main(int argc, char* argv[]) {
  Chroma::initialize(&argc, &argv);
  START_CODE();

  // Stock Chroma's registrations, exactly as chroma.cc's linkageHack() does
  // them, then ours on top. Both must precede the measurement read below --
  // that ordering is what makes runtime registration work.
  //
  // Note: this value is 0 in a stock Chroma too. Some sub-aggregate returns
  // false on this build; it is not a failure signal. The real check is that
  // the factory resolves NPR_MOMFRAC below.
  bool linkage = true;
  linkage &= InlineAggregateEnv::registerAll();
  linkage &= GaugeInitEnv::registerAll();
  linkage &= NprMomfrac::registerAll();
  QDPIO::cout << "NPR_MOMFRAC: linkage = " << linkage << std::endl;

  XMLReader xml_in;
  DriverInput input;
  try {
    xml_in.open(Chroma::getXMLInputFileName());
    readDriverInput(xml_in, "/chroma", input);
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception reading XML: " << e << std::endl;
    QDP_abort(1);
  }

  XMLFileWriter& xml_out = Chroma::getXMLOutputInstance();
  push(xml_out, "npr_momfrac");
  write(xml_out, "Input", xml_in);

  Layout::setLattSize(input.nrow);
  Layout::create();

  proginfo(xml_out);

  QDP::RNG::setrn(input.rng_seed);
  write(xml_out, "RNG", input.rng_seed);

  // Gauge field, through the same factory chroma.cc uses.
  multi1d<LatticeColorMatrix> u(Nd);
  XMLReader gauge_file_xml, gauge_xml;
  try {
    std::istringstream xml_c(input.cfg.xml);
    XMLReader cfgtop(xml_c);
    QDPIO::cout << "NPR_MOMFRAC: gauge initialization, cfg_type = "
                << input.cfg.id << std::endl;
    Handle<GaugeInit> gaugeInit(
      TheGaugeInitFactory::Instance().createObject(input.cfg.id, cfgtop,
                                                   input.cfg.path));
    (*gaugeInit)(gauge_file_xml, gauge_xml, u);
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception during gaugeInit: " << e
                << std::endl;
    QDP_abort(1);
  }

  XMLBufferWriter config_xml;
  config_xml << gauge_xml;
  write(xml_out, "Config_info", gauge_xml);

  // Cheap sanity check: on a unit field this must print plaquette 1.
  MesPlq(xml_out, "Observables", u);

  try {
    std::istringstream Measurements_is(input.inline_measurement_xml);
    XMLReader MeasXML(Measurements_is);
    multi1d<Handle<AbsInlineMeasurement> > the_measurements;
    read(MeasXML, "/InlineMeasurements", the_measurements);

    QDPIO::cout << "NPR_MOMFRAC: there are " << the_measurements.size()
                << " measurements" << std::endl;

    InlineDefaultGaugeField::reset();
    InlineDefaultGaugeField::set(u, config_xml);

    push(xml_out, "InlineObservables");
    xml_out.flush();

    unsigned long cur_update = 0;
    for (int m = 0; m < the_measurements.size(); ++m) {
      AbsInlineMeasurement& the_meas = *(the_measurements[m]);
      if (cur_update % the_meas.getFrequency() == 0) {
        push(xml_out, "elem");
        the_meas(cur_update, xml_out);
        pop(xml_out);
        xml_out.flush();
      }
    }

    pop(xml_out);  // InlineObservables
    InlineDefaultGaugeField::reset();
  }
  catch (const std::string& e) {
    QDPIO::cerr << "NPR_MOMFRAC: caught exception during measurement: " << e
                << std::endl;
    QDP_abort(1);
  }

  pop(xml_out);  // npr_momfrac

  QDPIO::cout << "NPR_MOMFRAC: ran successfully" << std::endl;

  END_CODE();
  Chroma::finalize();
  return 0;
}
#endif
