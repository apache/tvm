# CPU Sample Device API Implementation

## 概要

このディレクトリには、TVM用の`cpu_sample`デバイスAPIの基本実装が含まれています。これはCPU上でデバイスAPIの完全な動作を確認するためのサンプル実装です。

## 実装内容

### 1. デバイスタイプの定義

`include/tvm/runtime/device_api.h`に以下を追加:
- `kDLCPUSample = 20`: 新しいデバイスタイプの列挙値
- `DLDeviceType2Str`関数に`cpu_sample`のケースを追加

### 2. デバイスAPI実装

`src/runtime/cpu_sample_device_api.cc`:
- `CPUSampleDeviceAPI`クラス: `DeviceAPI`を継承
- 主要なメソッドの実装:
  - `SetDevice`: デバイスの設定
  - `GetAttr`: デバイス属性の取得
  - `AllocDataSpace`: メモリの割り当て
  - `FreeDataSpace`: メモリの解放
  - `CopyDataFromTo`: データのコピー
  - `AllocWorkspace` / `FreeWorkspace`: ワークスペースメモリの管理
  - `StreamSync`: ストリーム同期（CPU実装では空操作）

すべてのメソッドには`std::cout`でのログ出力が含まれており、動作を確認できます。

### 3. デバイスAPIの登録

`TVM_FFI_STATIC_INIT_BLOCK`を使用して、`device_api.cpu_sample`として登録されています。

## 使用方法

### ビルド

```bash
cd tvm
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Pythonでのテスト

```python
import tvm
import numpy as np

# cpu_sampleデバイスを作成 (device_type=20)
dev = tvm.device(20, 0)

# デバイスが存在するか確認
print(f"Device exists: {dev.exist}")

# 配列を割り当て
n = 1024
a = tvm.nd.array(np.random.randn(n).astype("float32"), dev)
print(f"Array allocated: {a.shape}")

# データをコピー
b = tvm.nd.array(np.zeros(n).astype("float32"), dev)
b.copyfrom(a)

# データを検証
a_np = a.numpy()
b_np = b.numpy()
print(f"Data match: {np.allclose(a_np, b_np)}")
```

### 自動テストスクリプト

提供されているテストスクリプトを実行:

```bash
cd tvm
python3 test_cpu_sample_device.py
```

## 期待される出力

テストを実行すると、以下のようなログ出力が表示されます:

```
[CPU_SAMPLE] Registering device_api.cpu_sample
[CPU_SAMPLE] SetDevice called: device_type=20, device_id=0
[CPU_SAMPLE] GetAttr called: device_type=20, device_id=0, kind=0
[CPU_SAMPLE] GetAttr kExist: returning 1
[CPU_SAMPLE] AllocDataSpace called: nbytes=4096, alignment=64
[CPU_SAMPLE] AllocDataSpace: allocated 4096 bytes at 0x...
...
```

## アーキテクチャ

```
┌─────────────────────────────────────┐
│  Python/C++ User Code               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  TVM Runtime API                    │
│  (Device, NDArray, etc.)            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  DeviceAPIManager                   │
│  - GetAPI("cpu_sample")             │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  CPUSampleDeviceAPI                 │
│  - SetDevice()                      │
│  - AllocDataSpace()                 │
│  - FreeDataSpace()                  │
│  - CopyDataFromTo()                 │
│  - etc.                             │
└─────────────────────────────────────┘
```

## 主な特徴

1. **完全なログ出力**: すべての主要なメソッド呼び出しでログを出力し、動作を追跡可能
2. **CPU互換性**: CPUメモリを使用した実装で、特別なハードウェア不要
3. **標準準拠**: TVMのDeviceAPIインターフェースに完全準拠
4. **拡張可能**: 新しいデバイスAPIを実装する際のテンプレートとして使用可能

## 技術的な詳細

### デバイスタイプ番号

- `kDLCPUSample = 20`: DLPack標準のデバイスタイプとの衝突を避けるため、20を選択
- `TVMDeviceExtType_End = 36`: TVMの拡張デバイスタイプの範囲内

### メモリ管理

- `posix_memalign`を使用したアライメント付きメモリ割り当て
- `WorkspacePool`を使用した効率的なワークスペース管理
- スレッドローカルストレージによる並列実行のサポート

### ストリーム管理

CPUデバイスのため、ストリーム操作は空実装（即座に完了とみなす）

## 今後の拡張

この基本実装をベースに、以下のような拡張が可能:

1. カスタムメモリアロケータの追加
2. パフォーマンスカウンタの統合
3. デバッグ/プロファイリング機能の追加
4. 他のデバイスタイプへの移植のテンプレートとして使用

## ファイル一覧

- `include/tvm/runtime/device_api.h`: デバイスタイプ定義
- `src/runtime/cpu_sample_device_api.cc`: デバイスAPI実装
- `test_cpu_sample_device.py`: テストスクリプト
- `CPU_SAMPLE_DEVICE_README.md`: このドキュメント

## ライセンス

Apache License 2.0 (TVMプロジェクトと同じ)
