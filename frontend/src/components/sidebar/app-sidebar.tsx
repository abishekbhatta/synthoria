import { User } from "lucide-react"
import { Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarMenu } from "../ui/sidebar"
import SidebarMenuItems from "./sidebar-menu-items"
import { UserButton } from "@daveyplate/better-auth-ui"
import { Credits } from "./credits"
import Upgrade from "./upgrade"


export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-primary mt-4 mb-12 flex flex-col items-start justify-start px-2 text-3xl font-black tracking-widest uppercase"><p>Synthoria</p></SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
                <SidebarMenuItems/>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    <SidebarFooter>
        <div className="mb-2 flex w-full items-center justify-center gap-1 text-xs">
          <Credits />
          <Upgrade />
        </div>
        <UserButton
          variant="outline"
          additionalLinks={[
            {
              label: "Customer Portal",
              href: "/customer-portal",
              icon: <User/>,
            },
          ]}
        />
      </SidebarFooter>
    </Sidebar>
  )
}