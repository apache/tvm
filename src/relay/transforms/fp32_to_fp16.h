#include <tvm/ir/op.h>
#include <tvm/relay/expr.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relay {

enum FP16ConversionCategory { RED, GRAY, GREEN };
std::unordered_map<FP16ConversionCategory, std::string> conversion_category_strings({{RED, "Red"},
                                                                                 {GRAY, "Gray"},
                                                                                 {GREEN, "Green"}});

using OpStringSet = std::unordered_set<std::string>;

// Default lists inspired from TF's classifications:
// github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
// They might have a bias toward NVidia's Tensor Cores so be aware and modify lists per your
// hardware choice.
OpStringSet DEFAULT_GREEN_LIST({
    "nn.conv1d",
    "nn.conv2d",
    "nn.conv3d",
    "nn.conv1d_transpose",
    "nn.conv2d_transpose",
    "nn.conv3d_transpose",
    "nn.dense",
});
OpStringSet DEFAULT_GRAY_LIST({
    // These ops add new data or change shape
    "nn.pad",
    "nn.batch_flatten",
    "concatenate",
    // Simple arithmetic
    "add",
    "nn.bias_add",
    "nn.batch_norm",
    // Simple activations
    "nn.relu",
    "nn.leaky_relu",
    "nn.prelu",
    "nn.dropout",
    // Pooling operations
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    // "nn.global_max_pool1d", // does not exist
    "nn.global_max_pool2d",
    // "nn.global_max_pool3d", // does not exist
    // "nn.global_avg_pool1d", // does not exist
    "nn.global_avg_pool2d",
    // "nn.global_avg_pool3d", // does not exist
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
});
OpStringSet DEFAULT_RED_LIST({
    // Activations with exponents or division
    "nn.cross_entropy",
    "nn.cross_entropy_with_logits",
    "nn.softmax",
    // Other
    "nn.l2_normalize",
});

class DefaultColorer {
 private:
  std::unordered_map<std::string, FP16ConversionCategory> op_to_initial_color;

 public:
  DefaultColorer(OpStringSet red_list = DEFAULT_RED_LIST, OpStringSet gray_list = DEFAULT_GRAY_LIST,
                 OpStringSet green_list = DEFAULT_GREEN_LIST) {
    std::vector<std::pair<OpStringSet, FP16ConversionCategory>> lists_and_colors{
        {red_list, RED}, {gray_list, GRAY}, {green_list, GREEN}};

    for (auto list_and_color : lists_and_colors) {
      OpStringSet ops = list_and_color.first;
      FP16ConversionCategory color = list_and_color.second;
      for (std::string op_name : ops) {
        op_to_initial_color.insert({{op_name, color}});
      }
    }
  }

  FP16ConversionCategory operator()(const tvm::relay::CallNode* call, bool ignore_missing = false) {
    auto* op_node = (call->op).as<tvm::OpNode>();
    if (op_node == nullptr) {
      throw std::invalid_argument("FP16 conversion only supports call nodes with op calls.");
    }

    std::string op_name = op_node->name;
    auto color = op_to_initial_color.find(op_name);

    if (color == op_to_initial_color.end()) {
      if (ignore_missing) {
        return RED;
      } else {
        throw std::invalid_argument("Op name " + op_name + " not in included lists!.");
      }
    }

    return color->second;
  }
};

}  // namespace relay
}  // namespace tvm