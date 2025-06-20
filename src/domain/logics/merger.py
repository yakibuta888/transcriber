from domain.common.progress_reporter import ProgressReporter


class ResultMerger:
    @staticmethod
    def merge(asr_chunks, diar_segments, progress: ProgressReporter | None = None):
        merged = []
        diar_segments = sorted(diar_segments, key=lambda x: x["start"])
        j = 0
        total = len(asr_chunks)

        if progress:
            progress.update.merge(0, "話者と文字起こしの結果を結合中...")

        for idx, chunk in enumerate(asr_chunks, 1):
            ws_start = chunk["timestamp"][0]
            ws_end = chunk["timestamp"][1]
            max_overlap = 0
            assigned_speaker = None
            while j < len(diar_segments) and diar_segments[j]["end"] < ws_start:
                j += 1
            for k in range(j, len(diar_segments)):
                seg = diar_segments[k]
                overlap = max(0, min(ws_end, seg["end"]) - max(ws_start, seg["start"]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = seg["speaker"]
                if seg["start"] > ws_end:
                    break
            merged.append({
                "speaker": assigned_speaker if assigned_speaker else "Unknown",
                "text": chunk["text"],
                "start": ws_start,
                "end": ws_end
            })
        # 進捗表示
        if progress and idx == total:
            progress.update.merge(idx, f"結合完了")
        elif progress:
            raise ValueError(f"Merge process did not complete as expected. It was ended {idx}/{total} processes. Check the input data. @ResultMerger.merge")
        return merged
