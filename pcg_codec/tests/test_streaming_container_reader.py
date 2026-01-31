from __future__ import annotations

import io
import struct

from pcg_codec.pcg.entropy.streams import (
    ContainerHeader,
    FramePacket,
    StreamingContainerReader,
    write_container,
)


def test_streaming_container_reader_header_only_chunk_does_not_buffererror() -> None:
    header = ContainerHeader(
        sample_rate_hz=48000,
        hop_samples=320,
        encoder_lookahead_samples=0,
        codec_name="pcg",
        params_hash="0" * 64,
    )
    frames = [
        FramePacket(frame_index=0, layer_bytes=[b"a", b"bc"]),
    ]
    buf = io.BytesIO()
    write_container(buf, header=header, frames=frames)
    container_bytes = buf.getvalue()

    header_len = struct.unpack_from("<I", container_bytes, 5)[0]
    header_end = 9 + int(header_len)

    reader = StreamingContainerReader()
    out0 = reader.feed(container_bytes[:header_end])
    assert out0 == []
    assert reader.header == header

    out1 = reader.feed(container_bytes[header_end : header_end + 1])
    assert out1 == []
    out2 = reader.feed(container_bytes[header_end + 1 :])
    assert len(out2) == 1
    assert out2[0].frame_index == 0
    assert out2[0].layer_bytes == [b"a", b"bc"]

