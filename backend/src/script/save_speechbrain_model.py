from speechbrain.inference.separation import SepformerSeparation

# 初回のみダウンロード＆ローカル保存
model = SepformerSeparation.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="src/models/mtl-mimic-voicebank"
)
