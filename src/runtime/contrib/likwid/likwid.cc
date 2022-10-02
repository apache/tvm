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


constexpr const char* REGION_NAME = "LikwidMetricCollector";


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
        likwid_markerRegisterRegion(REGION_NAME);
        likwid_markerStartRegion(REGION_NAME);
    }

    ObjectRef Start(Device device) override {
        likwid_markerThreadInit();
        int nevents = 20;
        double events[20];
        double time;
        int count;
        _read_event_counts(&nevents, events, &time, &count);
        std::vector<double> start_values(events, events + nevents * sizeof(double));
        return ObjectRef(make_object<LikwidEventSetNode>(start_values, device));
    }

    Map<String, ObjectRef> Stop(ObjectRef object) override {
        const LikwidEventSetNode* event_set_node = object.as<LikwidEventSetNode>();
        int nevents = 20;
        double events[20];
        double time;
        int count;
        _read_event_counts(&nevents, events, &time, &count);
        std::vector<double> end_values(events, events + nevents * sizeof(double));
        std::unordered_map<String, ObjectRef> reported_metrics;
        for (size_t i{}; i < nevents; ++i) {
            if (end_values[i] < event_set_node->start_values[i]) {
                LOG(WARNING) << "Detected overflow while reading performance counter, setting value to -1";
                reported_metrics[String(std::to_string(i))] = 
                    ObjectRef(make_object<CountNode>(-1));
            } else {
                reported_metrics[String(std::to_string(i))] = 
                    ObjectRef(make_object<CountNode>(end_values[i] - event_set_node->start_values[i]));
            }
        }
        return reported_metrics;
    }

    ~LikwidMetricCollectorNode() final {
        int res = likwid_markerStopRegion(REGION_NAME);
        if (res < 0) {
            LOG(ERROR) << "Could not stop marker region! Error code: " << res;
        }
        likwid_markerClose();
    }

    void _read_event_counts(int* nevents, double* events, double* time, int* count) {
        int status = likwid_markerStopRegion(REGION_NAME);
        if (status < 0) {
            LOG(ERROR) << "Could not stop marker region! Error code: " << status;
        }
        likwid_markerGetRegion(REGION_NAME, nevents, events, time, count);
        if (nevents == 0) {
            LOG(WARNING) << "Event count is zero!";
        }
        status = likwid_markerStartRegion(REGION_NAME);
        if (status < 0) {
            LOG(ERROR) << "Could not start marker region! Error code: " << status;
        }
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