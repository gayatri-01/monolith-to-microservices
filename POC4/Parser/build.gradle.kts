plugins {
    id("java")
}

group = "com.poc.parser"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    implementation("com.github.javaparser:javaparser-symbol-solver-core:3.25.8")
    implementation("org.json:json:20231013")

}

tasks.test {
    useJUnitPlatform()
}