#include <likwid.h>
#include <tvm/runtime/contrib/likwid.h>

#include <string>
#include <vector>

#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

namespace tvm {
namespace runtime {
namespace profiling {

struct LikwidEventSetNode : public Object {
    std::vector<double> start_values;
    Device dev;

    explicit LikwidEventSetNode(std::vector<double> start_values, Device dev) 
        : start_values(start_values), dev(dev) {}

    static constexpr const char* _type_key = "LikwidEventSetNode";
    TVM_DECLARE_FINAL_OBJECT_INFO(LikwidEventSetNode, Object);
};

struct LikwidMetricCollectorNode final : public MetricCollectorNode {
    explicit LikwidMetricCollectorNode(Array<DeviceWrapper> devices) {
        // Do nothing for now...
    }

    explicit LikwidMetricCollectorNode() {}

    void Init(Array<DeviceWrapper> devices) override {
        likwid_markerInit();
        likwid_markerThreadInit();
        likwid_markerRegisterRegion("MetricCollectorNode");
    }

    ObjectRef Start(Device device) override {
        int res = likwid_markerStartRegion("MetricCollectorNode");
        if (res < 0) {
            LOG(ERROR) << "Marker region could not be started! Error code: " << res;
            return ObjectRef(nullptr);
        }
        int nevents = 10;
        double events[10];
        double time;
        int count;
        likwid_markerGetRegion("LikwidEventSetNode", &nevents, events, &time, &count);
        if (count == 0) {
            LOG(WARNING) << "Event count is zero!";
        }
        std::vector<double> start_values(std::begin(events), std::end(events));
        return ObjectRef(make_object<LikwidEventSetNode>(start_values, device));
    }

    Map<String, ObjectRef> Stop(ObjectRef object) override {
        const LikwidEventSetNode* event_set_node = object.as<LikwidEventSetNode>();
        int nevents = 10;
        double events[10];
        double time;
        int count;
        likwid_markerGetRegion("LikwidEventSetNode", &nevents, events, &time, &count);
        std::vector<double> end_values(std::begin(events), std::end(events));
        std::unordered_map<String, ObjectRef> reported_metrics;
        for (size_t i{}; i < end_values.size(); ++i) {
            if (end_values[i] < event_set_node->start_values[i]) {
                LOG(WARNING) << "Detected overflow while reading performance counter, setting value to -1";
                reported_metrics[String(std::to_string(i))] = 
                    ObjectRef(make_object<CountNode>(-1));
            } else {
                reported_metrics[String(std::to_string(i))] = 
                    ObjectRef(make_object<CountNode>(end_values[i] - event_set_node->start_values[i]));
            }
        }
        int res = likwid_markerStopRegion("LikwidEventSetNode");
        if (res < 0) {
            LOG(ERROR) << "Could not close marker region! Error code: " << res;
        }
        return reported_metrics;
    }

    ~LikwidMetricCollectorNode() final {
        likwid_markerClose();
    }

    static constexpr const char* _type_key = "runtime.profiling.LikwidMetricCollector";
    TVM_DECLARE_FINAL_OBJECT_INFO(LikwidMetricCollectorNode, MetricCollectorNode);
};

class LikwidMetricCollector : public MetricCollector {
public:
    explicit LikwidMetricCollector(Array<DeviceWrapper> devices) {
        data_ = make_object<LikwidMetricCollectorNode>(devices);
    }
    TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LikwidMetricCollector, MetricCollector, 
                                          LikwidMetricCollectorNode);
};

TVM_REGISTER_OBJECT_TYPE(LikwidEventSetNode);
TVM_REGISTER_OBJECT_TYPE(LikwidMetricCollectorNode);

TVM_REGISTER_GLOBAL("runtime.profiling.LikwidMetricCollector")
    .set_body_typed([](Array<DeviceWrapper> devices) {
        return LikwidMetricCollector(devices);
    });

} // namespace profiling
} // namespace runtime
} // namespace tvm