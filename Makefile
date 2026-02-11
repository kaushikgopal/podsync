.PHONY: build clean test

build:
	cd scripts && cargo build --release
	cp scripts/target/release/podsync scripts/podsync

test:
	cd scripts && cargo test

clean:
	cd scripts && cargo clean
	rm -f scripts/podsync
