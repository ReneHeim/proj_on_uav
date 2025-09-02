#!/bin/bash
# install_deps.sh - Robust dependency installer with retry logic

set -e

echo "🔧 Installing dependencies with retry logic..."

# Configure pip for better reliability
pip config set global.timeout 120
pip config set global.retries 5

# Function to install with retries
install_with_retry() {
    local package="$1"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Installing $package"
        if pip install "$package" --timeout 120 --retries 3; then
            echo "✅ Successfully installed $package"
            return 0
        else
            echo "❌ Failed to install $package (attempt $attempt)"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "⏳ Waiting 10 seconds before retry..."
                sleep 10
            fi
        fi
    done
    
    echo "🚨 Failed to install $package after $max_attempts attempts"
    return 1
}

# Install main dependencies
echo "📦 Installing main dependencies..."
install_with_retry "-r /requirements.txt"

# Install essential test dependencies
echo "🧪 Installing test dependencies..."
install_with_retry "pytest==8.4.1"
install_with_retry "pytest-cov==6.2.1"

# Install additional dev tools (optional)
echo "🛠️ Installing additional dev tools (optional)..."
optional_packages=("black==25.1.0" "isort==6.0.1" "flake8==7.3.0" "mypy==1.17.1")

for package in "${optional_packages[@]}"; do
    if ! install_with_retry "$package"; then
        echo "⚠️ Optional package $package failed to install, continuing..."
    fi
done

echo "🎉 Dependency installation completed!"