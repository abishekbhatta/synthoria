"use client"

import { queueSong } from "~/actions/generation"
import { Button } from "./ui/button"

export default function CreateSong(){
    return <Button onClick={queueSong}>Generate Song</Button>
}