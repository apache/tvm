set -o errexit -o nounset
apt-get update && apt-get install -y openjdk-8-jdk maven
test -d "/usr/lib/jvm/java-8-openjdk-amd64/jre"
echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre" >> /etc/profile
