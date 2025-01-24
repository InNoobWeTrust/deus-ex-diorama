# DEUS-EX-DIORAMA

Just picked some cool random name for this experimental project, chosen the name for no reason other than "cool".


Purpose of this project is to have a full-stack monorepo that can be used to run GenAI and agents in multiple environments: CLI, desktop, web server, mobile, embedded, etc...


Kidding, it's too ambitious to say anything right now. This is an experimental project and I will drop it anytime I want, especially when I'm having a feeling of "out-of-money for a living". Maslow pyramid is a serious concern for those who want to build things alone without any team or obstacles from "professional and high-status persons"... 🥹


Poor dev here, doing this just to understand deeper about the building blocks of AI projects, so I can contribute to as many projects I can and earn bounty from there, hopefully...🖨️💵💵💵


## Quick start

### Install pkgx to setup development dependencies automatically (recommended)

```sh {"tag": "pkgx"}
curl -fsS https://pkgx.sh | sh
```

### Run a sample prompt

```sh {"tag": "make"}
## Install cargo-make using cargo or cargo-binstall
command -v cargo-make || \
    (command -v cargo-binstall && \
     cargo binstall cargo-make || \
     cargo install --no-default-features --force cargo-make)
## Or via cargo binstall
# cargo install --locked cargo-binstall
# cargo binstall cargo-make

## Invoke cargo subcommand
cargo make cli_simple
## Or call makers directly
# makers cli_simple
```

## Trying out the main rust library with a jupyter notebook

```sh {"tag": "evcxr"}
## Install evcxr_jupyter
command -v evcxr_jupyter || (command -v cargo-binstall && cargo binstall evcxr_jupyter || cargo install --locked evcxr_jupyter) && evcxr_jupyter --install
## Start jupyter lab from current directory
jupyter lab
```

Then choose `Rust` kernel to start a new notebook. For tutorials on how to use Rust for data analysis, see: [rust-data-analysis](https://github.com/wiseaidev/rust-data-analysis)

## Lazy?

Make sure you have `pkgx` and `runme` cli on your machine. Then you can run the shell code blocks in this README by referencing it with a tag.

For example, to try out the library with jupyter notebook as the guide above, just simply

```sh {"tag": "runme"}
runme run --filename README.md -t "evcxr" -y
```

## TODO

- [ ] Compile to Node-API lib (using [napi-rs](https://github.com/napi-rs/napi-rs)) for use with JS/TS frameworks (eg. [typechat](https://github.com/microsoft/TypeChat))
- [ ] Expose python API (using [pyo3](https://github.com/PyO3/pyo3)) for prompt-tuning locally or finetune models on Colab with less hacks (eg. [unsloth](https://github.com/unslothai/unsloth))
- [ ] WASM and WebGPU build for trying small models directly on browser (it's stupid but the UX is undeniably satisfying for those who are lazy AF to install anything and still want to "own their AI" 🤪 )
- [ ] Stop dreaming and admit that I can't finish all the TODOs this year, new year resolution for next year will be indeed refreshed with the year number only and no change to the content. 😃 
