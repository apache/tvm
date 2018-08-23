. /etc/profile

set -o errexit -o nounset

GRADLE_HOME=/opt/gradle
GRADLE_VERSION=4.10-rc-2
GRADLE_SHA256=e90d3c32910e259814bcca82b3911172ecca1ff1ab5ed69b4de3c1df8b378b40

echo "Downloading Gradle"
wget --output-document=gradle.zip "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"
echo "Checking Gradle hash"
echo "${GRADLE_SHA256} *gradle.zip" | sha256sum --check -
echo "Installing Gradle"
unzip gradle.zip
rm gradle.zip
mv "gradle-${GRADLE_VERSION}" "${GRADLE_HOME}/"
ln --symbolic "${GRADLE_HOME}/bin/gradle" /usr/bin/gradle
