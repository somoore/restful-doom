# Hellbox capsule for RESTful Doom agent mode.
FROM public.ecr.aws/lambda/microvms:al2023-minimal

ARG SDL2_VERSION=2.30.9
ARG SDL2_SHA256=24b574f71c87a763f50704bbb630cbe38298d544a1f890f099a4696b1d6beba4
ARG SDL2_MIXER_VERSION=2.8.0
ARG SDL2_MIXER_SHA256=1cfb34c87b26dbdbc7afd68c4f545c0116ab5f90bbfecc5aebe2a9cb4bb31549
ARG SDL2_NET_VERSION=2.2.0
ARG SDL2_NET_SHA256=4e4a891988316271974ff4e9585ed1ef729a123d22c08bd473129179dc857feb
ARG DOOM1_WAD_COMMIT=fcf22d9773c62835724f2d25c40933ef051f6600
ARG DOOM1_WAD_SHA256=1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771

RUN dnf install -y \
        autoconf automake binutils cargo gcc libtool make pkgconfig protobuf-compiler rust tar gzip \
        alsa-lib-devel pulseaudio-libs-devel which xz \
        ImageMagick xorg-x11-server-Xvfb \
        libX11-devel libXcursor-devel libXext-devel libXfixes-devel libXi-devel \
        libXinerama-devel libXrandr-devel libXrender-devel libXScrnSaver-devel \
        libXxf86vm-devel \
    && dnf clean all

WORKDIR /tmp/sdl
RUN curl -fsSL --retry 3 -o sdl.tgz "https://github.com/libsdl-org/SDL/releases/download/release-${SDL2_VERSION}/SDL2-${SDL2_VERSION}.tar.gz" \
    && printf '%s  %s\n' "$SDL2_SHA256" sdl.tgz | sha256sum -c - \
    && tar -xzf sdl.tgz --strip-components=1 \
    && ./configure --disable-video-wayland --disable-video-opengl \
        --disable-video-opengles --disable-video-vulkan --disable-sdltest \
    && make -j"$(nproc)" && make install \
    && echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig \
    && rm -rf /tmp/sdl

WORKDIR /tmp/mixer
RUN curl -fsSL --retry 3 -o mixer.tgz "https://github.com/libsdl-org/SDL_mixer/releases/download/release-${SDL2_MIXER_VERSION}/SDL2_mixer-${SDL2_MIXER_VERSION}.tar.gz" \
    && printf '%s  %s\n' "$SDL2_MIXER_SHA256" mixer.tgz | sha256sum -c - \
    && tar -xzf mixer.tgz --strip-components=1 \
    && ./configure --disable-music-mod-mikmod --disable-music-midi-fluidsynth \
        --disable-music-ogg --disable-music-flac --disable-music-mp3-mpg123 \
        --disable-music-opus --disable-sdltest \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/mixer

WORKDIR /tmp/net
RUN curl -fsSL --retry 3 -o net.tgz "https://github.com/libsdl-org/SDL_net/releases/download/release-${SDL2_NET_VERSION}/SDL2_net-${SDL2_NET_VERSION}.tar.gz" \
    && printf '%s  %s\n' "$SDL2_NET_SHA256" net.tgz | sha256sum -c - \
    && tar -xzf net.tgz --strip-components=1 \
    && ./configure --disable-sdltest \
    && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/net

COPY . /opt/restful-doom
WORKDIR /opt/restful-doom
RUN ACLOCAL_PATH=/usr/local/share/aclocal ./autogen.sh --disable-sdltest CFLAGS="-O2 -fcommon" \
    && make -j"$(nproc)" \
    && install -m 0755 src/restful-doom /usr/local/bin/restful-doom

RUN useradd -m -u 1000 app
COPY capsule/app/ /home/app/app/
RUN set -e; \
    if ! ls /home/app/app/*.wad /home/app/app/*.WAD >/dev/null 2>&1; then \
        echo "[build] fetching shareware DOOM1.WAD"; \
        curl -fsSL --retry 3 -o /home/app/app/DOOM1.WAD \
          "https://raw.githubusercontent.com/nneonneo/universal-doom/${DOOM1_WAD_COMMIT}/DOOM1.WAD"; \
        printf '%s  %s\n' "$DOOM1_WAD_SHA256" /home/app/app/DOOM1.WAD | sha256sum -c -; \
    fi; \
    chown -R app:app /home/app/app

COPY capsule/rootfs/ /
RUN chmod +x /opt/capsule/start.sh /opt/capsule/video_stream.py

EXPOSE 50051 6666 9000 6080
CMD ["/opt/capsule/start.sh"]
