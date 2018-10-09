/*!	
 * Copyright (c) 2018 by Contributors
 * \file graph_annotate.h
 * \brief Define rules to annotate a graph.
 */
 #ifndef NNVM_PASS_GRAPH_ANNOTATE_H_
 #define NNVM_PASS_GRAPH_ANNOTATE_H_

 #include <nnvm/graph.h>

 #include <string>
 #include <unordered_map>

 namespace nnvm {

 class ManualAnnotator;
 /*
  * This class is an abstract class that can be derived by other classes to
  * implement how a node should be selected.
  */
 class GraphAnnotator {
  public:
   explicit GraphAnnotator(int fallback_device)
       : fallback_device_(fallback_device) {}
   virtual ~GraphAnnotator() = default;
   // A virtual function that is implemented by different annotation methods.
   virtual int AnnotateNode(const nnvm::Node* n) const = 0;

   int GetFallbackDevice() const {
     return fallback_device_;
   }

  private:
   friend ManualAnnotator;
   /* The fallback device. */
   int fallback_device_;
 };

 /*
  * This class defines a manual way to annotate a graph node. In this method,
  * users are expected to provide the node name and also the device type that it
  * should be assigned to. However, if the operator contained in the graph node
  * is registered with a fallback property or the operator name has not been
  * saved, this node will be annotated with the fallback device.
  */
 class ManualAnnotator : public GraphAnnotator {
   using OpNameDeviceMap = std::unordered_map<std::string, int>;
  public:
   explicit ManualAnnotator(const OpNameDeviceMap& op_name_dev_map,
                            int fallback_device)
       : GraphAnnotator(fallback_device),
         op_name_dev_map_(new OpNameDeviceMap(op_name_dev_map)) {}

   int AnnotateNode(const nnvm::Node* n) const final {
     if (n->is_variable()) return 0;
     if (n->op()->fallback) return fallback_device_;

     return op_name_dev_map_->count(n->op()->name)
                ? op_name_dev_map_->at(n->op()->name)
                : fallback_device_;
   }

  private:
   std::unique_ptr<const OpNameDeviceMap> op_name_dev_map_;
 };

 using ManualAnnotatorPtr = std::shared_ptr<ManualAnnotator>;

 }  // namespace nnvm
 #endif  // NNVM_PASS_GRAPH_ANNOTATE_H_
